// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include "object-detection-post-processing.h"

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

class ObjectDetector_YoloV4Tiny
{
public:

    // Given an OpenVINO XML file path, and inference device ("CPU", "GPU", "VPUX", etc.),
    // create a Yolov4 Tiny Object Detector.
    ObjectDetector_YoloV4Tiny(std::string model_xml_file,
        std::string inference_device);

    // Given a BGR frame, perform inference and return a vector of objects.
    // This is called by our OBS plugin for each frame.
    std::vector<DetectedObject> Run(const cv::Mat& frameBGR);

private:
    ov::CompiledModel compiledModel;
    ov::InferRequest infer_request;
    ov::Layout yoloRegionLayout = "NCHW";

    std::shared_ptr< YoloV4Tiny_PostProcessor >  postProcessor;

    cv::Size inputTensorSize;

    std::string input_tensor_name;
    std::vector<std::string> outputTensorNames;
};
