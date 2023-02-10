// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <obs-module.h>
#include <media-io/video-scaler.h>
#include "plugin-macros.generated.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ittutils.h>
#include <obs_opencv_conversions.hpp>

#include "ovmediapipe/pose.h"
#include "ovmediapipe/selfie_segmentation.h"

enum class BackgroundConcealmentMethod
{
    UNSET,
    SELFIE_SEG_LANDSCAPE,
    POSE_LANDMARK_SEG,
};

struct ModelDetails
{
    std::string ui_display_name;
    BackgroundConcealmentMethod method;
};


static const std::list<ModelDetails> supported_models =
{
    {
        "Selfie Segmentation -- Landscape (MediaPipe)",
        BackgroundConcealmentMethod::SELFIE_SEG_LANDSCAPE,
    },
    {
        "Pose Landmark Segmentation (MediaPipe)",
        BackgroundConcealmentMethod::POSE_LANDMARK_SEG,
    }
};

struct background_concealment_filter {
    cv::Scalar backgroundColor{ 0, 0, 0 };
    int blur_value;
    bool blur;
    float smoothContour = 0.5f;
    bool background_image;

    std::string background_image_path;
    std::string Device;

    std::string modelSelection;
    BackgroundConcealmentMethod current_method;

    // Use the media-io converter to both scale and convert the colorspace
    video_scaler_t* scalerToBGR = nullptr;
    video_scaler_t* scalerFromBGR = nullptr;
    std::uint32_t nireq = 0;
    std::uint32_t nthreads = 0;
    std::string nstreams = "";
    std::string layout = "";
    bool auto_resize = false;

    cv::Mat blurred_img;

    std::vector<std::string> ov_available_devices;

    uint8_t segmask_threshold = 128;

    std::shared_ptr < ovmediapipe::Pose > pose;
    std::shared_ptr < ovmediapipe::SelfieSegmentation > selfie_seg;

    std::shared_ptr<cv::Mat> customImage;

};


static const char* filter_getname(void* unused)
{
    UNUSED_PARAMETER(unused);
    return "OpenVINO Background Concealment";
}


/**                   PROPERTIES                     */

static obs_properties_t* filter_properties(void* data)
{
    struct background_concealment_filter* tf = reinterpret_cast<background_concealment_filter*>(data);

    obs_properties_t* props = obs_properties_create();

    obs_property_t* p_model_select = obs_properties_add_list(
        props,
        "model_select",
        obs_module_text("Segmentation model"),
        OBS_COMBO_TYPE_LIST,
        OBS_COMBO_FORMAT_STRING);

    for (auto modentry : supported_models)
    {
        obs_property_list_add_string(p_model_select, obs_module_text(modentry.ui_display_name.c_str()), modentry.ui_display_name.c_str());
    }

    obs_properties_add_float_slider(
        props,
        "smooth_contour",
        obs_module_text("Smooth silhouette"),
        0.0,
        1.0,
        0.05);

    obs_properties_add_float_slider(
        props,
        "seg_mask_threshold",
        obs_module_text("Segmentation Mask Threshold"),
        0.0,
        1.0,
        0.05);

    obs_properties_add_color(
        props,
        "replaceColor",
        obs_module_text("Background Color"));

    obs_properties_add_bool(
        props,
        "AddCustomBackground",
        obs_module_text("Replace Background with custom image"));

    obs_properties_add_path(
        props,
        "CustomImagePath",
        obs_module_text("Custom Background Image Path"),
        OBS_PATH_FILE,
        ("*.jpeg" " * .jpg"),
        "");

    obs_properties_add_bool(
        props,
        "blur_background",
        obs_module_text("Background Blur"));

    obs_properties_add_float_slider(
        props,
        "blur_background_slider_val",
        obs_module_text("Background Blur Intensity"),
        0.0,
        1.0,
        0.05);

    obs_property_t* p_inf_device = obs_properties_add_list(
        props,
        "Device",
        obs_module_text("Inference device"),
        OBS_COMBO_TYPE_LIST,
        OBS_COMBO_FORMAT_STRING);

    for (auto device : tf->ov_available_devices)
    {
        if (device == "GNA")
            continue;

        obs_property_list_add_string(p_inf_device, obs_module_text(device.c_str()), device.c_str());
    }

    return props;
}

static void filter_defaults(obs_data_t* settings) {
    obs_data_set_default_double(settings, "smooth_contour", 0.5);
    obs_data_set_default_double(settings, "seg_mask_threshold", 0.5);
    obs_data_set_default_int(settings, "replaceColor", 0x00FF00);
    obs_data_set_default_bool(settings, "AddCustomBackground", false);
    obs_data_set_default_string(settings, "CustomImagePath", "");
    obs_data_set_default_bool(settings, "blur_background", true);
    obs_data_set_default_double(settings, "blur_background_slider_val", 0.5);
    obs_data_set_default_string(settings, "Device", "CPU");
    obs_data_set_default_string(settings, "model_select",
        supported_models.begin()->ui_display_name.c_str());

}

static void destroyScalers(struct background_concealment_filter* tf) {
    blog(LOG_INFO, "Destroy scalers.");
    if (tf->scalerToBGR != nullptr) {
        video_scaler_destroy(tf->scalerToBGR);
        tf->scalerToBGR = nullptr;
    }
    if (tf->scalerFromBGR != nullptr) {
        video_scaler_destroy(tf->scalerFromBGR);
        tf->scalerFromBGR = nullptr;
    }
}

#define SELFIE_SEG_LANDSCAPE_XML "../openvino-models/mediapipe_self_seg/FP16/selfie_segmentation_landscape.xml"
#define POSE_DETECTION_XML "../openvino-models/mediapipe_pose/pose_detection/FP16/pose_detection.xml"
#define POSE_LANDMARK_XML "../openvino-models/mediapipe_pose/pose_landmark/full/FP16/pose_landmark.xml"

#define MIN_BLUE_MASK_SIZE 5
#define MAX_BLUR_MASK_SIZE 41

static void filter_update(void* data, obs_data_t* settings)
{
    struct background_concealment_filter* tf = reinterpret_cast<background_concealment_filter*>(data);

    uint64_t color = obs_data_get_int(settings, "replaceColor");
    tf->backgroundColor.val[0] = (double)((color >> 16) & 0x0000ff);
    tf->backgroundColor.val[1] = (double)((color >> 8) & 0x0000ff);
    tf->backgroundColor.val[2] = (double)(color & 0x0000ff);

    tf->background_image = obs_data_get_bool(settings, "AddCustomBackground");

    std::string bkg_img_path = obs_data_get_string(settings, "CustomImagePath");
    if (tf->background_image && (tf->background_image_path != bkg_img_path))
    {
        try
        {
            tf->background_image_path = bkg_img_path;
            tf->customImage = std::make_shared<cv::Mat>(cv::imread(tf->background_image_path));
        }
        catch (const std::exception& e) {
            blog(LOG_ERROR, "%s", e.what());
            tf->customImage.reset();
            tf->background_image_path = "";
        }
    }

    tf->blur = obs_data_get_bool(settings, "blur_background");
    double blur_range_0_1 = obs_data_get_double(settings, "blur_background_slider_val"); 
    
    //remap it in the range [5,41]:
    tf->blur_value = (int)(blur_range_0_1 * (MAX_BLUR_MASK_SIZE - MIN_BLUE_MASK_SIZE) + MIN_BLUE_MASK_SIZE);

    tf->smoothContour = (float)obs_data_get_double(settings, "smooth_contour");
    tf->segmask_threshold = (uint8_t)(obs_data_get_double(settings, "seg_mask_threshold") * 255);

    const std::string current_device = obs_data_get_string(settings, "Device");
    const std::string newModel = obs_data_get_string(settings, "model_select");
    if (tf->modelSelection != newModel)
    {
        tf->selfie_seg.reset();
        tf->pose.reset();
    }

    BackgroundConcealmentMethod method = BackgroundConcealmentMethod::UNSET;
    for (auto modentry : supported_models)
    {
        if (modentry.ui_display_name == newModel)
        {
            method = modentry.method;
            break;
        }
    }


    if (method == BackgroundConcealmentMethod::SELFIE_SEG_LANDSCAPE && (!tf->selfie_seg || (tf->Device != current_device)))
    {
        tf->Device = current_device;
        tf->modelSelection = newModel;

        char* model_file_path = obs_module_file(SELFIE_SEG_LANDSCAPE_XML);
        if (model_file_path)
        {
            //create the pipeline
            try {
                blog(LOG_INFO, "Creating selfie segmentation, with model (%s)", model_file_path);
                auto selfie_seg = std::make_shared < ovmediapipe::SelfieSegmentation >(model_file_path, tf->Device);
                tf->selfie_seg = selfie_seg;
            }
            catch (const std::exception& e) {
                blog(LOG_ERROR, "%s", e.what());
            }
        }
        else
        {
            blog(LOG_ERROR, "Unable to find model file, %s, in obs-studio plugin module directory", SELFIE_SEG_LANDSCAPE_XML);
        }
    }
    else if (method == BackgroundConcealmentMethod::POSE_LANDMARK_SEG && (!tf->pose || (tf->Device != current_device)))
    {
        tf->Device = current_device;
        tf->modelSelection = newModel;

        char* pose_detection_xml = obs_module_file(POSE_DETECTION_XML);
        char* pose_landmark_xml = obs_module_file(POSE_LANDMARK_XML);
        if (pose_detection_xml && pose_landmark_xml)
        {
            try {
                blog(LOG_INFO, "Creating pose landmark segmentation, with models (%s), (%s)", pose_detection_xml, pose_landmark_xml);
                auto pose = std::make_shared < ovmediapipe::Pose >(pose_detection_xml, tf->Device, pose_landmark_xml, tf->Device);
                tf->pose = pose;
            }
            catch (const std::exception& e) {

                blog(LOG_ERROR, "%s", e.what());
            }
        }
        else
        {
            blog(LOG_ERROR, "Unable to find model files, %s & %sin obs-studio plugin module directory", POSE_DETECTION_XML, POSE_LANDMARK_XML);
        }
    }
}


/**                   FILTER CORE                     */

static void* filter_create(obs_data_t* settings, obs_source_t* source)
{
    background_concealment_filter* tf = new background_concealment_filter;

    ov::Core core;
    tf->ov_available_devices = core.get_available_devices();
    if (tf->ov_available_devices.empty())
    {
        blog(LOG_ERROR, "No available OpenVINO devices found.");
        delete tf;
        return NULL;
    }

    filter_update(tf, settings);

    return tf;
}


static void initializeScalers(
    cv::Size frameSize,
    enum video_format frameFormat,
    struct background_concealment_filter* tf
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

    blog(LOG_INFO, "Initialize scalers, from format %d to %d, and then back again Size %d x %d",
        frameFormat, VIDEO_FORMAT_BGR3, frameSize.width, frameSize.height);

    // Create new scalers
    video_scaler_create(&tf->scalerToBGR, &dst, &src, VIDEO_SCALE_DEFAULT);
    video_scaler_create(&tf->scalerFromBGR, &src, &dst, VIDEO_SCALE_DEFAULT);
}


static cv::Mat convertFrameToBGR(
    struct obs_source_frame* frame,
    struct background_concealment_filter* tf
) {
    ITT_SCOPED_TASK(bkg_rm_convertFrameToBGR);
    const cv::Size frameSize(frame->width, frame->height);

    if (tf->scalerToBGR == nullptr) {
        // Lazy initialize the frame scale & color converter
        initializeScalers(frameSize, frame->format, tf);
    }

    cv::Mat imageBGR(frameSize, CV_8UC3);
    if (obsframe_to_bgrmat(frame, imageBGR))
        return imageBGR;

    ITT_SCOPED_TASK(bkg_rm_convertFrameToBGR_video_scaler_scale);
    const uint32_t bgrLinesize = (uint32_t)(imageBGR.cols * imageBGR.elemSize());
    video_scaler_scale(tf->scalerToBGR,
        &(imageBGR.data), &(bgrLinesize),
        frame->data, frame->linesize);

    return imageBGR;
}

static void convertBGRToFrame(
    cv::Mat& imageBGR,
    struct obs_source_frame* frame,
    struct background_concealment_filter* tf
) {
    ITT_SCOPED_TASK(bkg_rm_convertBGRToFrame);
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

static void processImageForBackground(
    struct background_concealment_filter* tf,
    const cv::Mat& imageBGR,
    cv::Mat& backgroundMask)
{
    ITT_SCOPED_TASK(processImageForBackground);

    auto selfie_seg = tf->selfie_seg;
    auto pose = tf->pose;

    if (!selfie_seg && !pose)
    {
        backgroundMask = cv::Mat::zeros(imageBGR.size(), CV_8UC1);
        return;
    }

    try {
        cv::Mat outputImage;

        if (selfie_seg)
        {
            selfie_seg->Run(imageBGR, backgroundMask);
        }
        else
        {
            ovmediapipe::PoseLandmarkResult poselandmark_result;
            std::shared_ptr<cv::Mat> seg_mask = std::make_shared< cv::Mat >();
            pose->Run(imageBGR, poselandmark_result, seg_mask);
            backgroundMask = *seg_mask;
        }

        // Smooth mask with a fast filter (box).
        if (tf->smoothContour > 0.0) {
            //resize to 1/4 of frame size, apply gaussian blur (with potentially large kernel), upscale to frame size
            cv::Mat reduced_img;
            cv::resize(backgroundMask, reduced_img, cv::Size(imageBGR.cols / 4, imageBGR.rows / 4));
            int k_size = (int)(17 * tf->smoothContour);
            if ((k_size % 2) != 1)
                k_size++;
            cv::boxFilter(reduced_img, reduced_img, reduced_img.depth(), cv::Size(k_size, k_size));
            cv::resize(reduced_img, backgroundMask, cv::Size(imageBGR.cols, imageBGR.rows));
        }
        else
        {
            cv::resize(backgroundMask, backgroundMask, cv::Size(imageBGR.cols, imageBGR.rows));
        }

        backgroundMask = backgroundMask <= tf->segmask_threshold;
    }
    catch (const std::exception& e) {
        blog(LOG_ERROR, "%s", e.what());
    }
}

static struct obs_source_frame* filter_render(void* data, struct obs_source_frame* frame)
{
    ITT_SCOPED_TASK(bkg_rm_filter_render);
    struct background_concealment_filter* tf = reinterpret_cast<background_concealment_filter*>(data);

    // Convert to BGR
    cv::Mat imageBGR = convertFrameToBGR(frame, tf);

    cv::Mat backgroundMask(imageBGR.size(), CV_8UC1, cv::Scalar(255));

    // Process the image to find the mask.
    processImageForBackground(tf, imageBGR, backgroundMask);

    // Apply the mask back to the main image.
    try {
        if (tf->background_image || tf->blur)
        {
            if (tf->background_image)
            {
                ITT_SCOPED_TASK(apply_background_img);
                auto bkg_image = tf->customImage;
                if (bkg_image)
                {
                    if (bkg_image->size() != imageBGR.size())
                    {
                        cv::resize(*bkg_image, *bkg_image, imageBGR.size());
                    }

                    const int background_mask_cols = backgroundMask.cols;
                    for (int row = 0; row < backgroundMask.rows; ++row)
                    {
                        uint8_t* pMask = backgroundMask.ptr(row);
                        uint8_t* pBGR = imageBGR.ptr(row);
                        uint8_t* pCustomImage = bkg_image->ptr(row);

                        for (int col = 0; col < background_mask_cols; ++col) {
                            if (pMask[col])
                            {
                                pBGR[col * 3 + 0] = pCustomImage[col * 3 + 0];
                                pBGR[col * 3 + 1] = pCustomImage[col * 3 + 1];
                                pBGR[col * 3 + 2] = pCustomImage[col * 3 + 2];
                            }
                        }
                    }
                }
            }

            if (tf->blur) {

                //downscale, apply gaussian blur (with potentially large kernel), upscale
                cv::Mat reduced_img;
                cv::resize(imageBGR, reduced_img, cv::Size(imageBGR.cols / 4, imageBGR.rows / 4));
                //cv::boxFilter(imageBGR, tf->blurred_img, -1, cv::Size(tf->blur_value, tf->blur_value));
                //cv::boxFilter(reduced_img, reduced_img, -1, cv::Size(tf->blur_value, tf->blur_value));
                if ((tf->blur_value % 2) != 1)
                {
                    tf->blur_value++;
                }
                cv::GaussianBlur(reduced_img, reduced_img, cv::Size(tf->blur_value, tf->blur_value), 0.0);
                cv::resize(reduced_img, tf->blurred_img, imageBGR.size());
                
                {
                    ITT_SCOPED_TASK(blend_blurred);
                    const int background_mask_rows = backgroundMask.rows;
                    const int background_mask_cols = backgroundMask.cols;
                    cv::Mat backgroundMask3ch;
                    cv::cvtColor(backgroundMask, backgroundMask3ch, cv::COLOR_GRAY2BGR);
                    for (int row = 0; row < background_mask_rows; ++row)
                    {
                        const uint8_t* pMask = backgroundMask3ch.ptr(row);
                        const uint8_t* pBlurred = tf->blurred_img.ptr(row);
                        uint8_t* pBGR = imageBGR.ptr(row);
                        for (int col = 0; col < background_mask_cols * 3; ++col) 
                        {
                            pBGR[col] = (pBGR[col] & ~pMask[col]) | (pBlurred[col] & pMask[col]);
                        }
                    }
                }
            }
        }
        else
        {
            imageBGR.setTo(tf->backgroundColor, backgroundMask);
        }

    }
    catch (const std::exception& e) {
        blog(LOG_ERROR, "%s", e.what());
    }

    // Put masked image back on frame,
    convertBGRToFrame(imageBGR, frame, tf);
    return frame;
}


static void filter_destroy(void* data)
{
    struct background_concealment_filter* tf = reinterpret_cast<background_concealment_filter*>(data);

    if (tf) {
        destroyScalers(tf);
        delete tf;
    }
}



struct obs_source_info background_concealment_filter_info_ov = {
    .id = "background_concealment_ov",
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
