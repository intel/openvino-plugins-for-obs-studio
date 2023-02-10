// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "object-detection.h"

static inline ov::Layout getLayoutFromShape(const ov::Shape& shape) {
    if (shape.size() == 2) {
        return "NC";
    }
    else if (shape.size() == 3) {
        return (shape[0] >= 1 && shape[0] <= 4) ? "CHW" :
            "HWC";
    }
    else if (shape.size() == 4) {
        return (shape[1] >= 1 && shape[1] <= 4) ? "NCHW" :
            "NHWC";
    }
    else {
        throw std::runtime_error("Unsupported " + std::to_string(shape.size()) + "D shape");
    }
}

ObjectDetector_YoloV4Tiny::ObjectDetector_YoloV4Tiny(std::string model_xml_file,
    std::string inference_device)
{
    // Initialize OpenVINO Runtime Core 
    ov::Core core;

    // Read the model -
    // Note: We only give path to .xml file here, so the corresponsing 
    // .bin is assumed to reside in the same directory as the .xml (with the same 'base' name).
    std::shared_ptr<ov::Model> model = core.read_model(model_xml_file);

    // Given the ov::Model object, we can parse information from the network,
    // perform sanity checks (on expected number of input / outputs, etc.), as well as
    // set the format (layout, precision) for the inputs & outputs of the network.
    // 

    //We expect that this detection model takes a single input (the input image),
    // so here we're just double checking this..
    if (model->inputs().size() != 1) {
        throw std::logic_error("This plugin accepts a detection network that has only one input");
    }

    //Create a PrePostProcessor object, given the model. We'll use this to
    // set specifics around the format we will using for input, as well as the
    // expectations for output.
    ov::preprocess::PrePostProcessor ppp(model);

    // Since we will want to pass OpenCV BGR images as input to the network,
    // we want to set the network input (i.e. the input tensor) format,
    //  the precision and layout, to U8 (unsigned 8-bit) and NHWC (BGRBGRBGR)
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout({ "NHWC" });

    // Get the shape of model input, and from the shape we can determine what the
    // layout is via the helper function defined above, 'getLayoutFromShape'.
    const ov::Shape& inputShape = model->input().get_shape();
    ov::Layout inputLayout = getLayoutFromShape(inputShape);

    // Set the layout that we previously determined.
    ppp.input().model().set_layout(inputLayout);

    // Given a layout, we can extract channels, width, and height
    // from a 'Shape' object using the ov::layout::X_idx helper functions:
    // channels: ov::layout::channels_idx(inputLayout)
    // width: ov::layout::width_idx(inputLayout)
    // height: ov::layout::height_idx(inputLayout)

    // We expect an input shape is 3 channels
    if (inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("Expected 3-channel input");
    }

    //Cache the input tensor name. We'll use this later on.
    input_tensor_name = model->input().get_any_name();

    // Cache the input tensor width & height (from the input shape)
    inputTensorSize.height = inputShape[ov::layout::height_idx(inputLayout)];
    inputTensorSize.width = inputShape[ov::layout::width_idx(inputLayout)];

    // Output processing
    const ov::OutputVector& outputs = model->outputs();
    std::map<std::string, ov::Shape> outShapes;

    //For each output (there should be 2 here for Yolov4)
    for (auto& out : outputs) {

        //set precision & layout
        ppp.output(out.get_any_name()).tensor().set_element_type(ov::element::f32);
        ppp.output(out.get_any_name()).tensor().set_layout(yoloRegionLayout);

        //cache the output tensor names & shapes to use later
        outputTensorNames.push_back(out.get_any_name());
        outShapes[out.get_any_name()] = out.get_shape();
    }
    model = ppp.build();

    //sort the list of output tensor names by the corresponding output tensor height.
    std::sort(outputTensorNames.begin(),
        outputTensorNames.end(),
        [&outShapes, this](const std::string& x, const std::string& y) {
            return outShapes[x][ov::layout::height_idx(yoloRegionLayout)] >
                outShapes[y][ov::layout::height_idx(yoloRegionLayout)];
        });

    //create a vector of height sorted output shapes to initialize our
    // post-processor with.

    std::vector< std::vector<size_t> > height_sorted_nchw_output_shapes;
    for (const auto& name : outputTensorNames) {
        const auto& shape = outShapes[name];

        // The post-processor object expects the shapes to be passed in
        // 'NCHW' format, s
        std::vector<size_t> nchw_shape =
        { 1,
         shape[ov::layout::channels_idx(yoloRegionLayout)],
         shape[ov::layout::height_idx(yoloRegionLayout)],
         shape[ov::layout::width_idx(yoloRegionLayout)]
        };

        height_sorted_nchw_output_shapes.push_back(nchw_shape);
    }

    // We will be submitting 1 frame per inference, so this means a batch size of 1
    ov::set_batch(model, 1);

    // Generate the compiled model (the thing that will actually run on the specified) device,
    // from the model object. There's actually two things that happen here:
    // 1. Conversion from the 'generic' model representation (device agnostic) to a device-specific
    // format. In other words, the compilation step.
    // 2. The compiled executable is actually loaded onto the device.
    compiledModel = core.compile_model(model, inference_device);

    // From the compiled model object, create an InferRequest object. This is object that we will use to 
    // trigger inference (via infer() API). An InferRequest object contains the input & output tensor
    // memory buffers that are bound to a particular inference. Note that one could create multiple InferRequest objects to
    // trigger multiple inferences in parallel -- but we only create one here, as OBS is synchronous -- it
    // feeds us 1 frame at a time, and waits for the result before passing us the next frame.
    infer_request = compiledModel.create_infer_request();

    //create our pYolov4 Tiny post-processor helper object.
    postProcessor = std::make_shared< YoloV4Tiny_PostProcessor >(inputTensorSize.width,
        inputTensorSize.height, height_sorted_nchw_output_shapes);
}

std::vector<DetectedObject> ObjectDetector_YoloV4Tiny::Run(const cv::Mat& frameBGR)
{
    //******************************************************************
    //************ PRE-PROCESSING **************************************
    // *****************************************************************
    // In order to perform inference, we need to resize the input frame 
    // to the input tensor size.
    //
    // The infer_request object that we created during initialization
    // contains an already-allocated chunk of memory for this tensor, and we
    // can get access to that tensor object via the 'get_input_tensor' API
    ov::Tensor input_tensor = infer_request.get_input_tensor();

    // Get start pointer to the tensor data, as we will write to this.
    uint8_t* pTensor = input_tensor.data<uint8_t>();

    //pTensor is a pointer to the input tensor data. We want to 'wrap' this as an OpenCV Mat,
    // because that way we can resize our input frame directly *into* the tensor buffer. As opposed
    // to resizing it into a different OpenCV Mat object, and then copying the content into the tensor.
    cv::Mat input_tensor_wrapped_mat(inputTensorSize,
        CV_8UC3,
        pTensor);

    //perform resize from our full-size frame to our input tensor.
    cv::resize(frameBGR, input_tensor_wrapped_mat, inputTensorSize);

    // *****************************************
    // ******** RUN INFERENCE ******************
    // *****************************************
    // kick off inference, and wait for it to complete.
    infer_request.infer();

    // *****************************************
    // ******** POST PROCESSING ****************
    // *****************************************
    // Now that inference is complete, we need to convert the
    // data stored in the output tensors to something usable
    // (bounding boxes, labels, etc.).
    // We will use our yv4 tiny post-processor helper class for this.

    // Populate a vector of float ptr's, one for each output tensor (2 in this case)
    std::vector<const float*> out_tensors;
    for (const auto& outName : outputTensorNames)
    {
        ov::Tensor out_tensor = infer_request.get_tensor(outName);
        const float* pOutTensor = out_tensor.data<float>();
        out_tensors.push_back(pOutTensor);
    }

    //Invoke our helper objects post-proc function. Given the 'raw' tensor results, produce
    // detection results.
    std::vector<DetectedObject> objects = postProcessor->PostProcess(out_tensors, frameBGR.size());

    return objects;
}

