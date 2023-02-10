// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <obs-module.h>
#include <media-io/video-scaler.h>

#include "plugin-macros.generated.h"
#include <ittutils.h>
#include <obs_opencv_conversions.hpp>

#include "ovmediapipe/face_mesh.h"
#include "ovmediapipe/landmark_refinement_indices.h"

struct ModelDetails
{
    std::string ui_display_name;
    std::string data_path; //path to xml
};

static const std::list<ModelDetails> supported_face_detection_models =
{
    {
        "MediaPipe Face Detection (Full Range)",
        "../openvino-models/mediapipe_face_detection/full-range/FP16/face_detection-full_range-sparse-192x192.xml"
    },
    {
        "MediaPipe Face Detection (Short Range)",
        "../openvino-models/mediapipe_face_detection/short-range/FP16/face_detection_short_range-128x128.xml"
    },
};

static const std::list<ModelDetails> supported_face_landmark_models =
{
    {
        "MediaPipe Face Landmark",
        "../openvino-models/mediapipe_face_landmark/without_attention/FP16/face_landmark_without_attention_192x192.xml"
    },
    {
        "MediaPipe Face Landmark (With Attention)",
        "../openvino-models/mediapipe_face_landmark/with_attention/FP16/face_landmark_with_attention_192x192.xml"
    }
};

struct timestamped_roi
{
    uint64_t timestamp; //timestamp in ns
    cv::Rect roi;
};

struct face_effects_filter {
    // Use the media-io converter to both scale and convert the colorspace
    video_scaler_t* scalerToBGR = nullptr;
    video_scaler_t* scalerFromBGR = nullptr;

    bool bDebug_mode;

    std::string detectionModelSelection;
    std::string deviceDetectionInference;

    std::vector<std::string> ov_available_devices;

    std::shared_ptr < ovmediapipe::FaceMesh > facemesh;

    //face landmark
    std::string landmarkModelSelection;
    std::string deviceFaceMeshInference;
};

static const char* filter_getname(void* unused)
{
    UNUSED_PARAMETER(unused);
    return "OpenVINO Face Mesh";
}


/**                   PROPERTIES                     */
static obs_properties_t* filter_properties(void* data)
{
    struct face_effects_filter* tf = reinterpret_cast<face_effects_filter*>(data);

    obs_properties_t* props = obs_properties_create();

    obs_property_t* p_model_select = obs_properties_add_list(
        props,
        "model_select",
        obs_module_text("Face Detection Model"),
        OBS_COMBO_TYPE_LIST,
        OBS_COMBO_FORMAT_STRING);

    for (auto modentry : supported_face_detection_models)
    {
        obs_property_list_add_string(p_model_select, obs_module_text(modentry.ui_display_name.c_str()), modentry.ui_display_name.c_str());
    }

    obs_property_t* p_detection_inf_device = obs_properties_add_list(
        props,
        "DetectionInferenceDevice",
        obs_module_text("Face Detection Device"),
        OBS_COMBO_TYPE_LIST,
        OBS_COMBO_FORMAT_STRING);

    for (auto device : tf->ov_available_devices)
    {
        obs_property_list_add_string(p_detection_inf_device, obs_module_text(device.c_str()), device.c_str());
    }


    obs_property_t* p_landmark_model_select = obs_properties_add_list(
        props,
        "landmark_model_select",
        obs_module_text("Face Landmark Model"),
        OBS_COMBO_TYPE_LIST,
        OBS_COMBO_FORMAT_STRING);

    for (auto modentry : supported_face_landmark_models)
    {
        obs_property_list_add_string(p_landmark_model_select, obs_module_text(modentry.ui_display_name.c_str()), modentry.ui_display_name.c_str());
    }

    obs_property_t* p_facemesh_inf_device = obs_properties_add_list(
        props,
        "FaceMeshInferenceDevice",
        obs_module_text("Face Landmark Device"),
        OBS_COMBO_TYPE_LIST,
        OBS_COMBO_FORMAT_STRING);

    for (auto device : tf->ov_available_devices)
    {
        obs_property_list_add_string(p_facemesh_inf_device, obs_module_text(device.c_str()), device.c_str());
    }

    obs_properties_add_bool(props,
        "Debug-Mode",
        "Draw rectangle overlays to original frame");

    return props;
}

static void filter_defaults(obs_data_t* settings) {
    obs_data_set_default_bool(settings, "Debug-Mode", false);

    obs_data_set_default_string(settings, "model_select",
        supported_face_detection_models.begin()->ui_display_name.c_str());

    obs_data_set_default_string(settings, "landmark_model_select",
        supported_face_landmark_models.begin()->ui_display_name.c_str());

    obs_data_set_default_string(settings, "DetectionInferenceDevice", "CPU");
    obs_data_set_default_string(settings, "FaceMeshInferenceDevice", "CPU");
}

static void filter_update(void* data, obs_data_t* settings)
{
    struct face_effects_filter* tf = reinterpret_cast<face_effects_filter*>(data);

    const std::string current_detection_device = obs_data_get_string(settings, "DetectionInferenceDevice");
    const std::string current_detection_model = obs_data_get_string(settings, "model_select");

    const std::string current_facemesh_device = obs_data_get_string(settings, "FaceMeshInferenceDevice");
    const std::string current_landmark_model = obs_data_get_string(settings, "landmark_model_select");

    tf->bDebug_mode = obs_data_get_bool(settings, "Debug-Mode");


    if (!tf->facemesh ||
        (tf->deviceDetectionInference != current_detection_device) || (tf->detectionModelSelection != current_detection_model) ||
        (tf->deviceFaceMeshInference != current_facemesh_device) || (tf->landmarkModelSelection != current_landmark_model)
        )
    {
        tf->deviceDetectionInference = current_detection_device;
        tf->detectionModelSelection = current_detection_model;
        tf->deviceFaceMeshInference = current_facemesh_device;
        tf->landmarkModelSelection = current_landmark_model;
        blog(LOG_INFO, "Creating new Face Mesh Object. ");
        blog(LOG_INFO, "tf->deviceDetectionInference = %s ", tf->deviceDetectionInference.c_str());
        blog(LOG_INFO, "tf->deviceFaceMeshInference = %s ", tf->deviceFaceMeshInference.c_str());

        std::string model_data_path;
        for (auto modentry : supported_face_detection_models)
        {
            if (modentry.ui_display_name == current_detection_model)
            {
                model_data_path = modentry.data_path;
                break;
            }
        }

        std::string landmark_data_path;
        for (auto modentry : supported_face_landmark_models)
        {
            if (modentry.ui_display_name == current_landmark_model)
            {
                landmark_data_path = modentry.data_path;
                break;
            }
        }

        char* detection_model_file_path = obs_module_file(model_data_path.c_str());
        char* landmark_model_file_path = obs_module_file(landmark_data_path.c_str());

        if (detection_model_file_path && landmark_model_file_path)
        {
            try {
                auto facemesh = std::make_shared < ovmediapipe::FaceMesh >(detection_model_file_path, tf->deviceDetectionInference,
                    landmark_model_file_path, tf->deviceFaceMeshInference);
                tf->facemesh = facemesh;
            }
            catch (const std::exception& error) {
                blog(LOG_INFO, "in detection inference creation, exception: %s", error.what());
            }
        }
        else
        {
            blog(LOG_ERROR, "Could not find one of these required models: %s, %s", model_data_path.c_str(), landmark_data_path.c_str());
        }

    }
}

static void* filter_create(obs_data_t* settings, obs_source_t* source)
{
    face_effects_filter* tf = new face_effects_filter;

    ov::Core core;
    for (auto device : core.get_available_devices())
    {
        //don't allow GNA to be a supported device for this plugin.
        if (device == "GNA")
            continue;

        tf->ov_available_devices.push_back(device);
    }

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

static void destroyScalers(struct face_effects_filter* tf) {
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
    struct face_effects_filter* tf
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
    struct face_effects_filter* tf
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
    struct face_effects_filter* tf
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
    ITT_SCOPED_TASK(filter_render_face_mesh);
    struct face_effects_filter* tf = reinterpret_cast<face_effects_filter*>(data);

    try {
        auto facemesh = tf->facemesh;
        if (!facemesh)
            return frame;

        // Convert to BGR
        cv::Mat imageBGR = convertFrameToBGR(frame, tf);

        ovmediapipe::FaceLandmarksResults facelandmark_results;
        bool bDisplayResults = facemesh->Run(imageBGR, facelandmark_results);

        if (bDisplayResults)
        {
            ovmediapipe::RotatedRect roi = facelandmark_results.roi;
            roi.center_x *= imageBGR.cols;
            roi.center_y *= imageBGR.rows;
            roi.width *= imageBGR.cols;
            roi.height *= imageBGR.rows;

            for (int l = 0; l < 310; l++)
            {
                auto k = facelandmark_results.refined_landmarks[not_lips_eyes_indices[l]];
                int x = (int)(k.x * imageBGR.cols);
                int y = (int)(k.y * imageBGR.rows);

                //printf("keypoint = %d,%d\n", x, y);

                auto point = cv::Point(x, y);
                cv::circle(imageBGR, point, 2, cv::Scalar(0, 255, 0), -1);
            }

            for (int l = 0; l < 80; l++)
            {
                auto k = facelandmark_results.refined_landmarks[lips_refinement_indices[l]];
                int x = (int)(k.x * imageBGR.cols);
                int y = (int)(k.y * imageBGR.rows);

                //printf("keypoint = %d,%d\n", x, y);

                auto point = cv::Point(x, y);
                cv::circle(imageBGR, point, 2, cv::Scalar(0, 0, 255), -1);
            }

            for (int l = 0; l < 71; l++)
            {
                auto k = facelandmark_results.refined_landmarks[left_eye_refinement_indices[l]];
                int x = (int)(k.x * imageBGR.cols);
                int y = (int)(k.y * imageBGR.rows);

                //printf("keypoint = %d,%d\n", x, y);

                auto point = cv::Point(x, y);
                cv::circle(imageBGR, point, 2, cv::Scalar(255, 0, 0), -1);
            }

            for (int l = 0; l < 71; l++)
            {
                auto k = facelandmark_results.refined_landmarks[right_eye_refinement_indices[l]];
                int x = (int)(k.x * imageBGR.cols);
                int y = (int)(k.y * imageBGR.rows);

                //printf("keypoint = %d,%d\n", x, y);

                auto point = cv::Point(x, y);
                cv::circle(imageBGR, point, 2, cv::Scalar(0, 164, 255), -1);
            }

            if (facelandmark_results.refined_landmarks.size() > 468)
            {
                for (int l = 0; l < 5; l++)
                {
                    auto k = facelandmark_results.refined_landmarks[left_iris_refinement_indices[l]];
                    int x = (int)(k.x * imageBGR.cols);
                    int y = (int)(k.y * imageBGR.rows);

                    //printf("keypoint = %d,%d\n", x, y);

                    auto point = cv::Point(x, y);
                    cv::circle(imageBGR, point, 2, cv::Scalar(255, 255, 255), -1);
                }

                for (int l = 0; l < 5; l++)
                {
                    auto k = facelandmark_results.refined_landmarks[right_iris_refinement_indices[l]];
                    int x = (int)(k.x * imageBGR.cols);
                    int y = (int)(k.y * imageBGR.rows);

                    //printf("keypoint = %d,%d\n", x, y);

                    auto point = cv::Point(x, y);
                    cv::circle(imageBGR, point, 2, cv::Scalar(255, 255, 255), -1);
                }
            }
        }

        convertBGRToFrame(imageBGR, frame, tf);

    }
    catch (const std::exception& error) {
        blog(LOG_ERROR, "in OBS Face Mesh filter render: exception: %s", error.what());
    }
    catch (...) {
        blog(LOG_ERROR, " in OBS Face Mesh filter render: Unknown/internal exception happened");
    }

    return frame;
}

static void filter_destroy(void* data)
{
    struct face_effects_filter* tf = reinterpret_cast<face_effects_filter*>(data);

    if (tf) {
        destroyScalers(tf);

        delete tf;
    }
}

struct obs_source_info face_mesh_filter_info_ocv = {
    .id = "face_mesh_ov",
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
