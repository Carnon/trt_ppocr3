#include <iostream>
#include <vector>
#include <map>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "logging.h"
#include "common.hpp"

using namespace nvinfer1;


static Logger gLogger;

static const int INPUT_H = -1;
static const int INPUT_W = -1;
static const float scale = 0.5;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


int make_divisible(int v, int divisor=8){
    int new_v = std::max(divisor, int(v + divisor/2) / divisor * divisor);
    if(new_v < 0.9 * v) new_v += divisor;
    return new_v;
}


ICudaEngine* build_engine(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path){
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    INetworkDefinition* network = builder->createNetworkV2(1U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{1, 3, INPUT_H, INPUT_W});
    assert(data);

    ILayer* conv1 = convBnAct(network, weightMap, *data, 8, 3, 2, 1, 1, "hard_swish", "backbone.conv");

    // stage 0
    ILayer* conv2 = residualUnit(network, weightMap, *conv1->getOutput(0), make_divisible(int(16*scale)), make_divisible(int(16*scale)), 3, 1, false, true, "relu", "backbone.stages.0.0");
    ILayer* conv3 = residualUnit(network, weightMap, *conv2->getOutput(0), make_divisible(int(64*scale)), make_divisible(int(24*scale)), 3, 2, false, false, "relu", "backbone.stages.0.1");
    ILayer* conv4 = residualUnit(network, weightMap, *conv3->getOutput(0), make_divisible(int(72*scale)), make_divisible(int(24*scale)), 3, 1, false, true, "relu", "backbone.stages.0.2");

    // stage 1
    ILayer* conv5 = residualUnit(network, weightMap, *conv4->getOutput(0), make_divisible(int(72*scale)), make_divisible(int(40*scale)), 5, 2, false, false, "relu", "backbone.stages.1.0");
    ILayer* conv6 = residualUnit(network, weightMap, *conv5->getOutput(0), make_divisible(int(120*scale)), make_divisible(int(40*scale)), 5, 1, false, true, "relu", "backbone.stages.1.1");
    ILayer* conv7 = residualUnit(network, weightMap, *conv6->getOutput(0), make_divisible(int(120*scale)), make_divisible(int(40*scale)), 5, 1, false, true, "relu", "backbone.stages.1.2");

    // stage 2
    ILayer* conv8 = residualUnit(network, weightMap, *conv7->getOutput(0), make_divisible(int(240*scale)), make_divisible(int(80*scale)), 3, 2, false, false, "hard_swish", "backbone.stages.2.0");
    ILayer* conv9 = residualUnit(network, weightMap, *conv8->getOutput(0), make_divisible(int(200*scale)), make_divisible(int(80*scale)), 3, 1, false, true, "hard_swish", "backbone.stages.2.1");
    ILayer* conv10 = residualUnit(network, weightMap, *conv9->getOutput(0), make_divisible(int(184*scale)), make_divisible(int(80*scale)), 3, 1, false, true, "hard_swish", "backbone.stages.2.2");
    ILayer* conv11 = residualUnit(network, weightMap, *conv10->getOutput(0), make_divisible(int(184*scale)), make_divisible(int(80*scale)), 3, 1, false, true, "hard_swish", "backbone.stages.2.3");
    ILayer* conv12 = residualUnit(network, weightMap, *conv11->getOutput(0), make_divisible(int(480*scale)), make_divisible(int(112*scale)), 3, 1, false, false, "hard_swish", "backbone.stages.2.4");
    ILayer* conv13 = residualUnit(network, weightMap, *conv12->getOutput(0), make_divisible(int(672*scale)), make_divisible(int(112*scale)), 3, 1, false, true, "hard_swish", "backbone.stages.2.5");

    // stage 3
    ILayer* conv14 = residualUnit(network, weightMap, *conv13->getOutput(0), make_divisible(int(672*scale)), make_divisible(int(160*scale)), 5, 2, false, false, "hard_swish", "backbone.stages.3.0");
    ILayer* conv15 = residualUnit(network, weightMap, *conv14->getOutput(0), make_divisible(int(960*scale)), make_divisible(int(160*scale)), 5, 1, false, true, "hard_swish", "backbone.stages.3.1");
    ILayer* conv16 = residualUnit(network, weightMap, *conv15->getOutput(0), make_divisible(int(960*scale)), make_divisible(int(160*scale)), 5, 1, false, true, "hard_swish", "backbone.stages.3.2");
    ILayer* conv17 = convBnAct(network, weightMap, *conv16->getOutput(0), make_divisible(int(960*scale)), 1, 1, 0, 1, "hard_swish", "backbone.stages.3.3");

    // neck rsefpn
    IElementWiseLayer* in5 = rseLayer(network, weightMap, *conv17->getOutput(0), 96, 1, true, "neck.ins_conv.3");
    IElementWiseLayer* in4 = rseLayer(network, weightMap, *conv13->getOutput(0), 96, 1, true, "neck.ins_conv.2");
    IElementWiseLayer* in3 = rseLayer(network, weightMap, *conv7->getOutput(0), 96, 1, true, "neck.ins_conv.1");
    IElementWiseLayer* in2 = rseLayer(network, weightMap, *conv4->getOutput(0), 96, 1, true, "neck.ins_conv.0");

    float scale2[] = {1.0, 1.0, 2.0, 2.0};
    float scale4[] = {1.0, 1.0, 4.0, 4.0};
    float scale8[] = {1.0, 1.0, 8.0, 8.0};

    IResizeLayer* resize5 = network->addResize(*in5->getOutput(0));
    assert(resize5);
    resize5->setResizeMode(ResizeMode::kNEAREST);
    resize5->setScales(scale2, 4);
    IElementWiseLayer* out4 = network->addElementWise(*in4->getOutput(0), *resize5->getOutput(0), ElementWiseOperation::kSUM);

    IResizeLayer* resize4 = network->addResize(*out4->getOutput(0));
    assert(resize4);
    resize4->setResizeMode(ResizeMode::kNEAREST);
    resize4->setScales(scale2, 4);
    IElementWiseLayer* out3 = network->addElementWise(*in3->getOutput(0), *resize4->getOutput(0), ElementWiseOperation::kSUM);
    assert(out3);

    IResizeLayer* resize3 = network->addResize(*out3->getOutput(0));
    assert(resize3);
    resize3->setResizeMode(ResizeMode::kNEAREST);
    resize3->setScales(scale2, 4);
    IElementWiseLayer* out2 = network->addElementWise(*in2->getOutput(0), *resize3->getOutput(0), ElementWiseOperation::kSUM);

    IElementWiseLayer* p2 = rseLayer(network, weightMap, *out2->getOutput(0), 96/4, 3, true, "neck.inp_conv.0");
    IElementWiseLayer* p3 = rseLayer(network, weightMap, *out3->getOutput(0), 96/4, 3, true, "neck.inp_conv.1");
    IElementWiseLayer* p4 = rseLayer(network, weightMap, *out4->getOutput(0), 96/4, 3, true, "neck.inp_conv.2");
    IElementWiseLayer* p5 = rseLayer(network, weightMap, *in5->getOutput(0), 96/4, 3, true, "neck.inp_conv.3");

    IResizeLayer* rs5 = network->addResize(*p5->getOutput(0));
    rs5->setResizeMode(ResizeMode::kNEAREST);
    rs5->setScales(scale8, 4);

    IResizeLayer* rs4 = network->addResize(*p4->getOutput(0));
    rs4->setResizeMode(ResizeMode::kNEAREST);
    rs4->setScales(scale4, 4);

    IResizeLayer* rs3 = network->addResize(*p3->getOutput(0));
    rs3->setResizeMode(ResizeMode::kNEAREST);
    rs3->setScales(scale2, 4);

    ITensor* input_tensor_fuse[] = {rs5->getOutput(0), rs4->getOutput(0), rs3->getOutput(0), p2->getOutput(0)};
    IConcatenationLayer* fuse = network->addConcatenation(input_tensor_fuse, 4);
    fuse->setAxis(1);

    // head
    IConvolutionLayer* cv1 = network->addConvolutionNd(*fuse->getOutput(0), 96/4, DimsHW{3, 3}, weightMap["head.binarize.conv1.weight"], emptywts);
    cv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *cv1->getOutput(0), "head.binarize.conv_bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IDeconvolutionLayer* cv2 = network->addDeconvolutionNd(*relu1->getOutput(0), 96/4, DimsHW{2, 2}, weightMap["head.binarize.conv2.weight"], weightMap["head.binarize.conv2.bias"]);
    cv2->setStrideNd(DimsHW{2, 2});
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *cv2->getOutput(0), "head.binarize.conv_bn2", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    IDeconvolutionLayer* cv3 = network->addDeconvolutionNd(*relu2->getOutput(0), 1, DimsHW{2, 2}, weightMap["head.binarize.conv3.weight"], weightMap["head.binarize.conv3.bias"]);
    assert(cv3);
    cv3->setStrideNd(DimsHW{2, 2});

    IActivationLayer* sig1 = network->addActivation(*cv3->getOutput(0), ActivationType::kSIGMOID);

    sig1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*sig1->getOutput(0));

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1, 3, 96, 96));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(1, 3, 512, 512));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(1, 3, 2048, 2048));
    config->addOptimizationProfile(profile);

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize((1<<30));
    config->setFlag(BuilderFlag::kFP16);

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return engine;
}


void APIToModel(IHostMemory** modelStream, std::string &wts_path) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = build_engine(builder, config, DataType::kFLOAT, wts_path);

    assert(engine != nullptr);
    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}


int main() {

    cudaSetDevice(0);
    char *trtModelStream{ nullptr };
    size_t size{0};

    IHostMemory* modelStream{nullptr};
    std::string wts_path = "../../../weights/ch_ptocr_v3_det_infer.wts";

    APIToModel(&modelStream, wts_path);
    assert(modelStream != nullptr);
    std::ofstream p("../../../weights/ch_ptocr_v3_det_infer.engine", std::ios::binary);
    if(!p){
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    modelStream->destroy();
//    std::cout<<make_divisible(12)<<std::endl;
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
