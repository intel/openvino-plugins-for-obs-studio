// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "ovmediapipe/selfie_segmentation.h"
#include "ovmediapipe/common.h"

namespace ovmediapipe
{
    SelfieSegmentation::SelfieSegmentation(std::string model_xml_path, std::string device)
    {
        //TODO: Move this to some singleton-like ovmediapipe object
        ov::Core core;

        std::shared_ptr<ov::Model> model = core.read_model(model_xml_path);
        logBasicModelInfo(model);

        //prepare inputs / outputs
        {
            if (model->inputs().size() != 1) {
                throw std::logic_error("SelfieSegmentation model topology should have only 1 input");
            }

            const auto& input = model->input();
            const ov::Shape& inputShape = model->input().get_shape();
            ov::Layout inputLayout = getLayoutFromShape(input.get_shape());

            if (inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
                throw std::logic_error("Expected 3-channel input");
            }

            ov::preprocess::PrePostProcessor ppp(model);
            ppp.input().tensor().set_element_type(ov::element::u8).set_layout({ "NHWC" });

            ppp.input().model().set_layout(inputLayout);

            inputsNames.push_back(model->input().get_any_name());
            netInputWidth = inputShape[ov::layout::width_idx(inputLayout)];
            netInputHeight = inputShape[ov::layout::height_idx(inputLayout)];

            const ov::OutputVector& outputs = model->outputs();

            if (outputs.size() != 1)
                throw std::range_error("Expected model with 2 outputs");

            for (auto& out : outputs) {
                ppp.output(out.get_any_name()).tensor().set_element_type(ov::element::f32);
                outputsNames.push_back(out.get_any_name());
            }

            model = ppp.build();
        }

        ov::set_batch(model, 1);

        compiledModel = core.compile_model(model, device);
        inferRequest = compiledModel.create_infer_request();
    }

    void SelfieSegmentation::Run(const cv::Mat& frameBGR, cv::Mat& seg_mask)
    {
        preprocess(frameBGR);

        // perform inference 
        inferRequest.infer();

        postprocess(frameBGR, seg_mask);
    }

    void SelfieSegmentation::preprocess(const cv::Mat& frameBGR)
    {
        const ov::Tensor& frameTensor = inferRequest.get_tensor(inputsNames[0]);  // first input should be image
        uint8_t* pTensor = frameTensor.data<uint8_t>();

        //wrap the already-allocated tensor as a cv::Mat
        cv::Mat transformed = cv::Mat(netInputHeight, netInputWidth, CV_8UC3, pTensor);

        cv::resize(frameBGR, transformed, cv::Size(netInputWidth, netInputHeight), cv::INTER_LINEAR);
    }

    void SelfieSegmentation::postprocess(const cv::Mat& frameBGR, cv::Mat& seg_mask)
    {
        const auto data = inferRequest.get_tensor(outputsNames[0]).data<float>();

        //double check that the output tensor have the correct size
        {
            size_t raw_data_size = inferRequest.get_tensor(outputsNames[0]).get_byte_size();
            if (raw_data_size < (netInputHeight * netInputWidth * sizeof(float)))
            {
                throw std::logic_error("output tensor is holding a smaller amount of data than expected.");
            }
        }

        cv::Mat fg_confidence = cv::Mat(netInputHeight, netInputWidth, CV_32FC1, data);
        fg_confidence.convertTo(seg_mask, CV_8UC1, 255.0);
    }

} // namespace ovmediapipe