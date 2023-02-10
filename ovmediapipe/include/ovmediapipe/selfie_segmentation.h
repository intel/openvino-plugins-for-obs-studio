// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

namespace ovmediapipe
{
    class SelfieSegmentation
    {
    public:
        SelfieSegmentation(std::string model_xml_path, std::string device);

        void Run(const cv::Mat& frameBGR, cv::Mat& seg_mask);

    private:

        void preprocess(const cv::Mat& frameBGR);
        void postprocess(const cv::Mat& frameBGR, cv::Mat& seg_mask);

        std::vector<std::string> inputsNames;
        std::vector<std::string> outputsNames;

        ov::InferRequest inferRequest;
        ov::CompiledModel compiledModel;

        size_t netInputHeight = 0;
        size_t netInputWidth = 0;
    };
}