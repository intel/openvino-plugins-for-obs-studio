// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

class YoloV4Tiny_PostProcessorImpl;

struct DetectedObject : public cv::Rect2f {
    unsigned int labelID;
    std::string label;
    float confidence;
};

class YoloV4Tiny_PostProcessor
{
public:

    YoloV4Tiny_PostProcessor(unsigned long netInputWidth,
        unsigned long netInputHeight,
        std::vector< std::vector<size_t> >& height_sorted_nchw_output_shapes);

    //Given a vector of pointers to output tensors, and original frame size, generate
    // detection results (vector of bounding boxes)
    std::vector<DetectedObject> PostProcess(std::vector<const float*>& out_tensors,
        cv::Size origFrameSize);

private:
    std::shared_ptr <YoloV4Tiny_PostProcessorImpl> m_pImp;
};
