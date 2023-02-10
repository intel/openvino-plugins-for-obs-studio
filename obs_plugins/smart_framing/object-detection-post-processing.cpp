// Copyright(C) 2022-2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "object-detection-post-processing.h"
#include <ngraph/ngraph.hpp>

std::vector<float> yolov4_tiny_defaultAnchors =
{ 10.0f, 14.0f, 23.0f, 27.0f, 37.0f, 58.0f,
  81.0f, 82.0f, 135.0f, 169.0f, 344.0f, 319.0f };

const std::vector<int64_t> yolov4_tiny_defaultMasks =
{ 1, 2, 3, 3, 4, 5 };

static inline float sigmoid(float x) {
    return 1.f / (1.f + exp(-x));
}
static inline float linear(float x) {
    return x;
}

int calculateEntryIndex(int totalCells, int lcoords, int lclasses, int location, int entry) {
    int n = location / totalCells;
    int loc = location % totalCells;
    return (n * (lcoords + lclasses + 1) + entry) * totalCells + loc;
}

double intersectionOverUnion(const DetectedObject& o1, const DetectedObject& o2) {
    double overlappingWidth = fmin(o1.x + o1.width, o2.x + o2.width) - fmax(o1.x, o2.x);
    double overlappingHeight = fmin(o1.y + o1.height, o2.y + o2.height) - fmax(o1.y, o2.y);
    double intersectionArea = (overlappingWidth < 0 || overlappingHeight < 0) ? 0 : overlappingHeight * overlappingWidth;
    double unionArea = o1.width * o1.height + o2.width * o2.height - intersectionArea;
    return intersectionArea / unionArea;
}

class Region {
public:
    int num = 0;
    int classes = 0;
    int coords = 0;
    std::vector<float> anchors;
    int outputWidth = 0;
    int outputHeight = 0;

    Region(const std::shared_ptr<ov::op::v0::RegionYolo>& regionYolo);
    Region(int classes, int coords, const std::vector<float>& anchors, const std::vector<int64_t>& masks, int outputWidth, int outputHeight);
};

Region::Region(const std::shared_ptr<ov::op::v0::RegionYolo>& regionYolo) {
    coords = regionYolo->get_num_coords();
    classes = regionYolo->get_num_classes();
    auto mask = regionYolo->get_mask();
    num = mask.size();

    auto shape = regionYolo->get_input_shape(0);
    outputWidth = shape[3];
    outputHeight = shape[2];

    if (num) {

        // Parsing YoloV3 parameters
        anchors.resize(num * 2);

        for (int i = 0; i < num; ++i) {
            anchors[i * 2] = regionYolo->get_anchors()[mask[i] * 2];
            anchors[i * 2 + 1] = regionYolo->get_anchors()[mask[i] * 2 + 1];
        }
    }
    else {

        throw std::logic_error("Expected mask.size() to be something >0");
    }
}

Region::Region(int classes, int coords, const std::vector<float>& anchors, const std::vector<int64_t>& masks, int outputWidth, int outputHeight) :
    classes(classes), coords(coords),
    outputWidth(outputWidth), outputHeight(outputHeight) {
    num = masks.size();

    if (anchors.size() == 0 || anchors.size() % 2 != 0) {
        throw std::runtime_error("Explicitly initialized region should have non-empty even-sized regions vector");
    }

    if (num) {
        this->anchors.resize(num * 2);

        for (int i = 0; i < num; ++i) {
            this->anchors[i * 2] = anchors[masks[i] * 2];
            this->anchors[i * 2 + 1] = anchors[masks[i] * 2 + 1];
        }
    }
    else {
        this->anchors = anchors;
        num = anchors.size() / 2;
    }
}

class YoloV4Tiny_PostProcessorImpl
{
public:
    YoloV4Tiny_PostProcessorImpl(unsigned long nInputWidth,
        unsigned long nInputHeight,
        std::vector< std::vector<size_t> >& h_sorted_output_shapes)
        : netInputWidth(nInputWidth), netInputHeight(nInputHeight)
    {
        if (h_sorted_output_shapes.size() != 2)
        {
            throw std::logic_error("This post-processor expects h_sorted_output_shapes to contain exactly 2 shapes");
        }

        for (auto& out_shape : h_sorted_output_shapes) {
            if (out_shape.size() != 4)
                throw std::logic_error("This post-processor expects each output shape to have 4 dimensions");
        }

        int num = 3;
        int i = 0;

        auto chosenMasks = yolov4_tiny_defaultMasks;

        for (auto& shape : h_sorted_output_shapes) {
            if (shape[1] % num != 0) {
                throw std::runtime_error(std::string("The output shape index ") + std::to_string(i) + " has wrong 2nd dimension");
            }
            auto classes = (shape[1] / num) - 5;

            h_sorted_regions.emplace_back(Region(classes, 4,
                yolov4_tiny_defaultAnchors,
                std::vector<int64_t>(chosenMasks.begin() + i * num, chosenMasks.begin() + (i + 1) * num),
                shape[3], shape[2]));
            i++;
        }
    }

    std::vector<DetectedObject> PostProcess(std::vector<const float*>& out_tensors,
        cv::Size origFrameSize)
    {
        std::vector<DetectedObject> objects;
        int i = 0;
        for (const float* pOutTensor : out_tensors)
        {
            this->parseYOLOOutput(pOutTensor, h_sorted_regions[i], origFrameSize.height, origFrameSize.width, objects);
            i++;
        }

        // Advanced postprocessing
        // Checking IOU threshold conformance
        // For every i-th object we're finding all objects it intersects with, and comparing confidence
        // If i-th object has greater confidence than all others, we include it into result
        std::vector<DetectedObject> results;
        for (const auto& obj1 : objects) {
            bool isGoodResult = true;
            for (const auto& obj2 : objects) {
                if (obj1.labelID == obj2.labelID && obj1.confidence < obj2.confidence && intersectionOverUnion(obj1, obj2) >= boxIOUThreshold) { // if obj1 is the same as obj2, condition expression will evaluate to false anyway
                    isGoodResult = false;
                    break;
                }
            }
            if (isGoodResult) {
                results.push_back(obj1);
            }
        }

        return results;
    }

    void parseYOLOOutput(const float* output_blob,
        const Region& region,
        const unsigned long original_im_h,
        const unsigned long original_im_w,
        std::vector<DetectedObject>& objects)
    {
        int sideW = 0;
        int sideH = 0;
        unsigned long scaleH;
        unsigned long scaleW;

        sideH = region.outputHeight;
        sideW = region.outputWidth;
        scaleW = netInputWidth;
        scaleH = netInputHeight;

        auto entriesNum = sideW * sideH;

        auto postprocessRawData = sigmoid;

        // --------------------------- Parsing YOLO Region output -------------------------------------
        for (int i = 0; i < entriesNum; ++i) {
            int row = i / sideW;
            int col = i % sideW;
            for (int n = 0; n < region.num; ++n) {
                //--- Getting region data from blob
                int obj_index = calculateEntryIndex(entriesNum, region.coords, region.classes, n * entriesNum + i, region.coords);
                int box_index = calculateEntryIndex(entriesNum, region.coords, region.classes, n * entriesNum + i, 0);
                float scale = postprocessRawData(output_blob[obj_index]);

                //--- Preliminary check for confidence threshold conformance
                if (scale >= confidenceThreshold) {
                    //--- Calculating scaled region's coordinates
                    double x = (col + postprocessRawData(output_blob[box_index + 0 * entriesNum])) / sideW * original_im_w;
                    double y = (row + postprocessRawData(output_blob[box_index + 1 * entriesNum])) / sideH * original_im_h;
                    double height = std::exp(output_blob[box_index + 3 * entriesNum]) * region.anchors[2 * n + 1] * original_im_h / scaleH;
                    double width = std::exp(output_blob[box_index + 2 * entriesNum]) * region.anchors[2 * n] * original_im_w / scaleW;

                    DetectedObject obj;
                    obj.x = (float)std::max((x - width / 2), 0.);
                    obj.y = (float)std::max((y - height / 2), 0.);
                    obj.width = std::min((float)width, original_im_w - obj.x);
                    obj.height = std::min((float)height, original_im_h - obj.y);

                    for (int j = 0; j < region.classes; ++j) {
                        int class_index = calculateEntryIndex(entriesNum, region.coords, region.classes, n * entriesNum + i, region.coords + 1 + j);
                        float prob = scale * postprocessRawData(output_blob[class_index]);

                        //--- Checking confidence threshold conformance and adding region to the list
                        if (prob >= confidenceThreshold) {
                            obj.confidence = prob;
                            obj.labelID = j;
                            obj.label = getLabelName(obj.labelID);
                            objects.push_back(obj);
                        }
                    }
                }
            }
        }
    }

private:

    std::vector<Region> h_sorted_regions;
    float confidenceThreshold = 0.5f;
    float boxIOUThreshold = 0.5;
    std::vector<std::string> labels;

    unsigned long netInputWidth;
    unsigned long netInputHeight;
    
    std::string getLabelName(int labelID) { return (size_t)labelID < labels.size() ? labels[labelID] : std::string("Label #") + std::to_string(labelID); }
};

YoloV4Tiny_PostProcessor::YoloV4Tiny_PostProcessor(unsigned long netInputWidth,
    unsigned long netInputHeight,
    std::vector< std::vector<size_t> >& h_sorted_output_shapes)
{
    m_pImp = std::make_shared< YoloV4Tiny_PostProcessorImpl >(netInputWidth,
        netInputHeight,
        h_sorted_output_shapes);
}

std::vector<DetectedObject> YoloV4Tiny_PostProcessor::PostProcess(std::vector<const float*>& out_tensors,
    cv::Size origFrameSize)
{
    return m_pImp->PostProcess(out_tensors, origFrameSize);
}
