// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <obs-module.h>
#include <media-io/video-scaler.h>

#include "plugin-macros.generated.h"
#include <ittutils.h>
#include <obs_opencv_conversions.hpp>

#include "object-detection.h"
#include "ovmediapipe/face_detection.h"

struct ModelDetails
{
    std::string ui_display_name;
    std::string data_path; //path to xml
};

static const std::list<ModelDetails> supported_models =
{
    {
        "MediaPipe Face Detection (Full Range)",
        "../openvino-models/mediapipe_face_detection/full-range/FP16/face_detection-full_range-sparse-192x192.xml"
    },
    {
        "MediaPipe Face Detection (Short Range)",
        "../openvino-models/mediapipe_face_detection/short-range/FP16/face_detection_short_range-128x128.xml"
    },
    {
        "Yolo-v4 Tiny",
        "../openvino-models/yolo-v4-tiny/tf/FP16-INT8/yolo-v4-tiny.xml"
    }
};

struct timestamped_roi
{
    uint64_t timestamp; //timestamp in ns
    cv::Rect roi;
};

struct smart_framing_filter {
    // Use the media-io converter to both scale and convert the colorspace
    video_scaler_t* scalerToBGR = nullptr;
    video_scaler_t* scalerFromBGR = nullptr;

    int sum_x = 0;
    int sum_y = 0;
    int sum_w = 0;
    int sum_h = 0;
    std::list<timestamped_roi> roi_list;

    bool bDebug_mode;
    bool bSmoothMode;
    int smooth_duration_secs;

    std::string detectionModelSelection;
    std::string deviceDetectionInference;

    std::vector<std::string> ov_available_devices;

    ov::Core core;

    //OpenVINO objects
    std::shared_ptr< ObjectDetector_YoloV4Tiny > yv4_tiny_detection;

    std::shared_ptr < ovmediapipe::FaceDetection > face_detection;

    float zoom = 1.0f;
};

static const char* filter_getname(void* unused)
{
    UNUSED_PARAMETER(unused);
    return "OpenVINO Smart Framing";
}


/**                   PROPERTIES                     */
static obs_properties_t* filter_properties(void* data)
{
    struct smart_framing_filter* tf = reinterpret_cast<smart_framing_filter*>(data);

    obs_properties_t* props = obs_properties_create();

    obs_property_t* p_model_select = obs_properties_add_list(
        props,
        "model_select",
        obs_module_text("Detection Model"),
        OBS_COMBO_TYPE_LIST,
        OBS_COMBO_FORMAT_STRING);

    for (auto modentry : supported_models)
    {
        obs_property_list_add_string(p_model_select, obs_module_text(modentry.ui_display_name.c_str()), modentry.ui_display_name.c_str());
    }

    obs_properties_add_bool(props,
        "Debug-Mode",
        "Draw rectangle overlays to original frame");

    obs_properties_add_bool(props,
        "Smooth",
        "Smooth");

    obs_properties_add_float_slider(
        props,
        "Zoom",
        obs_module_text("Zoom"),
        0.0,
        2.0,
        0.25);

    obs_properties_add_int_slider(
        props,
        "SmoothDuration",
        obs_module_text("Smooth Region-of-Interest Duration (in seconds)"),
        1,
        4,
        1);

    obs_property_t* p_detection_inf_device = obs_properties_add_list(
        props,
        "DetectionInferenceDevice",
        obs_module_text("Inference Device for detection"),
        OBS_COMBO_TYPE_LIST,
        OBS_COMBO_FORMAT_STRING);


    for (auto device : tf->ov_available_devices)
    {
        if (device == "GNA")
            continue;

        obs_property_list_add_string(p_detection_inf_device, obs_module_text(device.c_str()), device.c_str());
    }

    return props;
}

static void filter_defaults(obs_data_t* settings) {
    obs_data_set_default_bool(settings, "Debug-Mode", false);

    obs_data_set_default_string(settings, "model_select",
        supported_models.begin()->ui_display_name.c_str());

    obs_data_set_default_bool(settings, "Smooth", true);
    obs_data_set_default_int(settings, "SmoothDuration", 2);
    obs_data_set_default_double(settings, "Zoom", 1.0);
    obs_data_set_default_string(settings, "DetectionInferenceDevice", "CPU");
}

static void filter_update(void* data, obs_data_t* settings)
{
    struct smart_framing_filter* tf = reinterpret_cast<smart_framing_filter*>(data);

    const std::string current_detection_device = obs_data_get_string(settings, "DetectionInferenceDevice");

    tf->bDebug_mode = obs_data_get_bool(settings, "Debug-Mode");
    tf->bSmoothMode = obs_data_get_bool(settings, "Smooth");
    tf->smooth_duration_secs = (int)obs_data_get_int(settings, "SmoothDuration");

    tf->zoom = (float)obs_data_get_double(settings, "Zoom");

    const std::string current_detection_model = obs_data_get_string(settings, "model_select");

    if (tf->detectionModelSelection != current_detection_model)
    {
        blog(LOG_INFO, "destroying instances");
        tf->yv4_tiny_detection.reset();
        tf->face_detection.reset();
    }

    std::string model_data_path;
    for (auto modentry : supported_models)
    {
        if (modentry.ui_display_name == current_detection_model)
        {
            model_data_path = modentry.data_path;
            break;
        }
    }

    char* model_file_path = obs_module_file(model_data_path.c_str());
    if (!model_file_path)
    {
        blog(LOG_ERROR, "Could not find required model (%s) in this plugins data directory", model_data_path.c_str());
        return;
    }

    if ((current_detection_model == "Yolo-v4 Tiny") && (!tf->yv4_tiny_detection || (tf->deviceDetectionInference != current_detection_device) || (tf->detectionModelSelection != current_detection_model)))
    {
        tf->deviceDetectionInference = current_detection_device;
        tf->detectionModelSelection = current_detection_model;

        blog(LOG_INFO, "Filter update: Creating new Detection Inference Object. Device = %s", tf->deviceDetectionInference.c_str());
        blog(LOG_INFO, "Model used = %s", tf->detectionModelSelection.c_str());

        blog(LOG_INFO, "Model File Path = %s", model_file_path);

        try {
            std::shared_ptr< ObjectDetector_YoloV4Tiny > detector =
                std::make_shared< ObjectDetector_YoloV4Tiny >(model_file_path,
                    tf->deviceDetectionInference);
            tf->yv4_tiny_detection = detector;
        }
        catch (const std::exception& error) {
            blog(LOG_INFO, "in detection inference creation, exception: %s", error.what());
        }
    }
    else if ((!tf->face_detection || (tf->deviceDetectionInference != current_detection_device) || (tf->detectionModelSelection != current_detection_model)))
    {
        blog(LOG_INFO, "Filter update: Creating new Face Detection Object. Device = %s", tf->deviceDetectionInference.c_str());

        tf->deviceDetectionInference = current_detection_device;
        tf->detectionModelSelection = current_detection_model;

        try {
            auto face_detection = std::make_shared < ovmediapipe::FaceDetection >(model_file_path, tf->deviceDetectionInference, tf->core, 1.f);
            tf->face_detection = face_detection;
        }
        catch (const std::exception& error) {
            blog(LOG_INFO, "in face detection creation, exception: %s", error.what());
        }

    }
}

static void* filter_create(obs_data_t* settings, obs_source_t* source)
{
    smart_framing_filter* tf = new smart_framing_filter;


    tf->ov_available_devices = tf->core.get_available_devices();
    if (tf->ov_available_devices.empty())
    {
        blog(LOG_INFO, "No available OpenVINO devices found.");
        delete tf;
        return NULL;
    }

    /** Configure networks **/
    filter_update(tf, settings);

    return tf;
}

static void destroyScalers(struct smart_framing_filter* tf) {
    if (tf->scalerToBGR != nullptr) {
        video_scaler_destroy(tf->scalerToBGR);
        tf->scalerToBGR = nullptr;
    }
    if (tf->scalerFromBGR != nullptr) {
        video_scaler_destroy(tf->scalerFromBGR);
        tf->scalerFromBGR = nullptr;
    }
}

static void initializeScalers(
    cv::Size frameSize,
    enum video_format frameFormat,
    struct smart_framing_filter* tf
) {

    struct video_scale_info dst {
        VIDEO_FORMAT_BGR3,
            (uint32_t)frameSize.width,
            (uint32_t)frameSize.height,
            VIDEO_RANGE_DEFAULT,
            VIDEO_CS_DEFAULT
    };
    struct video_scale_info src {
        frameFormat,
            (uint32_t)frameSize.width,
            (uint32_t)frameSize.height,
            VIDEO_RANGE_DEFAULT,
            VIDEO_CS_DEFAULT
    };

    // Check if scalers already defined and release them
    destroyScalers(tf);

    blog(LOG_INFO, "Initialize scalers. Size %d x %d",
        frameSize.width, frameSize.height);

    // Create new scalers
    video_scaler_create(&tf->scalerToBGR, &dst, &src, VIDEO_SCALE_DEFAULT);
    video_scaler_create(&tf->scalerFromBGR, &src, &dst, VIDEO_SCALE_DEFAULT);
}


static cv::Mat convertFrameToBGR(
    struct obs_source_frame* frame,
    struct smart_framing_filter* tf
) {
    ITT_SCOPED_TASK(convertFrameToBGR);
    const cv::Size frameSize(frame->width, frame->height);

    if (tf->scalerToBGR == nullptr) {
        // Lazy initialize the frame scale & color converter
        initializeScalers(frameSize, frame->format, tf);
    }


    cv::Mat imageBGR(frameSize, CV_8UC3);
    if (obsframe_to_bgrmat(frame, imageBGR))
        return imageBGR;

    const uint32_t bgrLinesize = (uint32_t)(imageBGR.cols * imageBGR.elemSize());
    video_scaler_scale(tf->scalerToBGR,
        &(imageBGR.data), &(bgrLinesize),
        frame->data, frame->linesize);

    return imageBGR;
}

static void convertBGRToFrame(
    cv::Mat& imageBGR,
    struct obs_source_frame* frame,
    struct smart_framing_filter* tf
) {
    ITT_SCOPED_TASK(convertBGRToFrame);
    if (tf->scalerFromBGR == nullptr) {
        // Lazy initialize the frame scale & color converter
        initializeScalers(cv::Size(frame->width, frame->height), frame->format, tf);
    }

    if (bgrmat_to_obsframe(imageBGR, frame))
        return;

    const uint32_t rgbLinesize = (uint32_t)(imageBGR.cols * imageBGR.elemSize());
    video_scaler_scale(tf->scalerFromBGR,
        frame->data, frame->linesize,
        &(imageBGR.data), &(rgbLinesize));
}

static struct obs_source_frame* filter_render(void* data, struct obs_source_frame* frame)
{
    ITT_SCOPED_TASK(filter_render_smart_framing);
    struct smart_framing_filter* tf = reinterpret_cast<smart_framing_filter*>(data);

    try {
        std::vector<DetectedObject> objects;
        auto yv4_tiny_detection = tf->yv4_tiny_detection;
        auto face_detection = tf->face_detection;
        if (!yv4_tiny_detection && !face_detection)
        {
            blog(LOG_INFO, "Error in detection inference creation");
            return frame;
        }

        // Convert to BGR
        cv::Mat imageBGR = convertFrameToBGR(frame, tf);

        if (yv4_tiny_detection)
        {
            objects = yv4_tiny_detection->Run(imageBGR);
        }
        else
        {
            std::vector<ovmediapipe::DetectedObject> face_objects;
            face_detection->Run(imageBGR, face_objects);

            for (auto fo : face_objects) {
                DetectedObject obj;
                obj.confidence = fo.confidence;
                obj.x = fo.x * imageBGR.cols;
                obj.y = fo.y * imageBGR.rows;
                obj.width = fo.width * imageBGR.cols;
                obj.height = fo.height * imageBGR.rows;
                obj.labelID = 0;
                objects.push_back(obj);
            }
        }

        cv::Rect init_rect;
        for (const auto& el : objects) {
            if (el.labelID == 0) {//person ID
                cv::Rect obj = static_cast<cv::Rect>(el);
                float size_multiplier;
                float adjust_top_multiplier;
                if (tf->detectionModelSelection != "Yolo-v4 Tiny")
                {
                    //normalize in the range [2.0 - 3.0]
                    size_multiplier = (1.0f - tf->zoom / 2.0f) + 2.0f;
                    adjust_top_multiplier = 0.75f;
                }
                else
                {
                    //normalize in the range [0.75 - 1.25]
                    size_multiplier = (1.0f - tf->zoom / 2.0f) * (1.25f - 0.75f) + 0.75f;
                    adjust_top_multiplier = 1.0f;
                }

                //adjust height
                {
                    float new_height = obj.height * size_multiplier;
                    if (new_height > (float)imageBGR.size().height)
                        new_height = (float)imageBGR.size().height;

                    if ((new_height - obj.height) > 0)
                    {
                        obj.y -= (int)((new_height - obj.height) / (2.0f * adjust_top_multiplier));
                        if (obj.y < 0)
                        {
                            obj.y = 0;
                        }
                    }
                    obj.height = (int)new_height;


                    if (obj.y + obj.height > imageBGR.size().height) {
                        obj.y = imageBGR.size().height - obj.height;
                    }
                }

                //adjust width
                {
                    float new_width = obj.width * size_multiplier;
                    if (new_width > (float)imageBGR.size().width)
                        new_width = (float)imageBGR.size().width;

                    obj.x -= (int)((new_width - obj.width) / 2.0f);
                    if (obj.x < 0)
                    {
                        obj.x = 0;
                    }
                    obj.width = (int)new_width;

                    if (obj.x + obj.width > imageBGR.size().width) {
                        obj.x = imageBGR.size().width - obj.width;
                    }
                }


                init_rect = init_rect | obj;
            }
        }

        //very unlikely, but detect a timestamp 'roll-over'
        if (!tf->roi_list.empty() && tf->roi_list.back().timestamp > frame->timestamp)
        {
            tf->roi_list.clear();
            tf->sum_x = 0;
            tf->sum_y = 0;
            tf->sum_w = 0;
            tf->sum_h = 0;
        }

        //pop off all timestamps in our list that are not within our duration.
        uint64_t time_threshold = 1000000000ULL * tf->smooth_duration_secs;
        if (frame->timestamp > time_threshold)
        {
            while (!tf->roi_list.empty())
            {
                auto timestamped_roi = tf->roi_list.front();
                if (timestamped_roi.timestamp < (frame->timestamp - time_threshold))
                {
                    cv::Rect old_roi = tf->roi_list.front().roi;
                    tf->sum_x -= old_roi.x;
                    tf->sum_y -= old_roi.y;
                    tf->sum_w -= old_roi.width;
                    tf->sum_h -= old_roi.height;
                    tf->roi_list.pop_front();
                }
                else
                {
                    break;
                }
            }
        }

        //if we didn't find anyone in the scene, set our 'init_rect' to the entire scene area.
        if (init_rect.empty())
        {
            init_rect.x = 0;
            init_rect.y = 0;
            init_rect.width = frame->width;
            init_rect.height = frame->height;
        }

        tf->roi_list.push_back(timestamped_roi{ frame->timestamp, init_rect });
        tf->sum_x += init_rect.x;
        tf->sum_y += init_rect.y;
        tf->sum_w += init_rect.width;
        tf->sum_h += init_rect.height;

        cv::Rect avg_roi;
        avg_roi.x = tf->sum_x / tf->roi_list.size();
        avg_roi.y = tf->sum_y / tf->roi_list.size();
        avg_roi.width = tf->sum_w / tf->roi_list.size();
        avg_roi.height = tf->sum_h / tf->roi_list.size();


        cv::Rect the_roi;
        if (tf->bSmoothMode)
        {
            the_roi = avg_roi;
        }
        else
        {
            the_roi = init_rect;
        }

        if ((uint32_t)the_roi.width == frame->width && (uint32_t)the_roi.height == frame->height)
        {
            return frame;
        }

        std::string debug_label_base = "Person ";
        if (tf->detectionModelSelection != "Yolo-v4 Tiny")
        {
            debug_label_base = "Face ";
        }

        //calculate cropped region, and perform crop + resize
        {
            //calulcate adjusted ROI
            cv::Rect adjusted_rect;

            float aspect_ratio_adjusted_height = ((static_cast<float>(imageBGR.size().height) *
                static_cast<float>(the_roi.width)) /
                (static_cast<float>(imageBGR.size().width)));

            if (aspect_ratio_adjusted_height < the_roi.height)
            {
                adjusted_rect.y = the_roi.y;
                adjusted_rect.height = the_roi.height;
                adjusted_rect.width = static_cast<int>((static_cast<float>(imageBGR.size().width) *
                    static_cast<float>(the_roi.height)) /
                    (static_cast<float>(imageBGR.size().height)));
                int x_delta = adjusted_rect.width - the_roi.width;
                int even_x_delta = (x_delta % 2 == 0) ? (x_delta) : (x_delta - 1);
                adjusted_rect.x = the_roi.x - (even_x_delta / 2);

                //collision with left side of scene
                if (adjusted_rect.x < 0) {
                    adjusted_rect.x = 0;
                }

                //collision with right side of scene
                if (adjusted_rect.x + adjusted_rect.width > imageBGR.size().width) {
                    adjusted_rect.x = imageBGR.size().width - adjusted_rect.width;
                }
            }
            else
            {
                adjusted_rect.x = the_roi.x;
                adjusted_rect.width = the_roi.width;
                adjusted_rect.height = static_cast<int>((static_cast<float>(imageBGR.size().height) *
                    static_cast<float>(the_roi.width)) /
                    (static_cast<float>(imageBGR.size().width)));
                int y_delta = adjusted_rect.height - the_roi.height;
                int even_y_delta = (y_delta % 2 == 0) ? (y_delta) : (y_delta - 1);
                adjusted_rect.y = the_roi.y - (even_y_delta / 2);

                //collision with top of scene
                if (adjusted_rect.y < 0) {
                    adjusted_rect.y = 0;
                }

                //collision with bottom of scene
                if (adjusted_rect.y + adjusted_rect.height > imageBGR.size().height) {
                    adjusted_rect.y = imageBGR.size().height - adjusted_rect.height;
                }
            }

            if (tf->bDebug_mode)
            {
                //in debug mode, just overlay ROI's onto original frame, instead
                // of performing 'real' crop / resize.
                cv::rectangle(imageBGR, adjusted_rect, cv::Scalar{ 0,255,255 }, 3, cv::LINE_8, 0);
                cv::rectangle(imageBGR, the_roi, cv::Scalar{ 255,0,0 }, 3, cv::LINE_8, 0);

                //Draw detections on original image
                int person_idx = 0;
                for (const auto& el : objects) {
                    if (el.labelID == 0) //personId
                    {
                        cv::rectangle(imageBGR, el, cv::Scalar{ 0,255,0 }, 2, cv::LINE_8, 0);
                        cv::putText(imageBGR, debug_label_base + std::to_string(person_idx++) + " " + std::to_string((int)el.width) + "x" + std::to_string((int)el.height), el.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, { 0,255,0 }, 2);
                    }
                }
            }
            else
            {
                //crop
                cv::Mat SF_ROI;
                imageBGR(adjusted_rect).copyTo(SF_ROI);

                cv::resize(SF_ROI, imageBGR, imageBGR.size());
            }
        }

        convertBGRToFrame(imageBGR, frame, tf);

    }
    catch (const std::exception& error) {
        blog(LOG_ERROR, "in OBS Smart Framing filter render: exception: %s", error.what());
    }
    catch (...) {
        blog(LOG_ERROR, " in OBS Smart Framing filter render: Unknown/internal exception happened");
    }

    return frame;
}

static void filter_destroy(void* data)
{
    struct smart_framing_filter* tf = reinterpret_cast<smart_framing_filter*>(data);

    if (tf) {
        destroyScalers(tf);

        delete tf;
    }
}

struct obs_source_info smart_framing_filter_info_ocv = {
    .id = "smart_framing_ov",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
    .get_name = filter_getname,
    .create = filter_create,
    .destroy = filter_destroy,
    .get_defaults = filter_defaults,
    .get_properties = filter_properties,
    .update = filter_update,
    .filter_video = filter_render,
};
