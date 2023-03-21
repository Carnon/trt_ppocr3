#include <iostream>
#include <vector>
#include <map>
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "logging.h"
#include "common.hpp"

using namespace nvinfer1;


static Logger gLogger;

static const int INPUT_H = 48;
static const int INPUT_W = -1;
static const int NUM_CLASS = 6625;
static const float scale = 0.5;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


ICudaEngine* build_engine(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path){
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(1U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{1, 3, INPUT_H, INPUT_W});
    assert(data);

    // backbone MobileNetV1Enhance
    IElementWiseLayer* conv1 = convBnHardSwish(network, weightMap, *data, int(32.f*scale), 3, 2, 2, 1, 1, "backbone.conv1");
    IElementWiseLayer* conv2_1 = depthWiseSeparable(network, weightMap, *conv1->getOutput(0), 32, 64, 32, 1, 1, scale, 3, 1,false, "backbone.block_list.0");
    IElementWiseLayer* conv2_2 = depthWiseSeparable(network, weightMap, *conv2_1->getOutput(0), 64, 128, 64, 1, 1, scale, 3, 1,false, "backbone.block_list.1");
    IElementWiseLayer* conv3_1 = depthWiseSeparable(network, weightMap, *conv2_2->getOutput(0), 128, 128, 128, 1, 1, scale, 3, 1,false, "backbone.block_list.2");
    IElementWiseLayer* conv3_2 = depthWiseSeparable(network, weightMap, *conv3_1->getOutput(0), 128, 256, 128, 2, 1, scale, 3, 1, false, "backbone.block_list.3");
    IElementWiseLayer* conv4_1 = depthWiseSeparable(network, weightMap, *conv3_2->getOutput(0), 256, 256, 256, 1, 1, scale, 3, 1, false, "backbone.block_list.4");
    IElementWiseLayer* conv4_2 = depthWiseSeparable(network, weightMap, *conv4_1->getOutput(0), 256, 512, 256, 2, 1, scale, 3, 1, false, "backbone.block_list.5");

    IElementWiseLayer* conv5_1 = depthWiseSeparable(network, weightMap, *conv4_2->getOutput(0), 512, 512, 512, 1, 1, scale, 5, 2, false, "backbone.block_list.6");
    IElementWiseLayer* conv5_2 = depthWiseSeparable(network, weightMap, *conv5_1->getOutput(0), 512, 512, 512, 1, 1, scale, 5, 2, false, "backbone.block_list.7");
    IElementWiseLayer* conv5_3 = depthWiseSeparable(network, weightMap, *conv5_2->getOutput(0), 512, 512, 512, 1, 1, scale, 5, 2, false, "backbone.block_list.8");
    IElementWiseLayer* conv5_4 = depthWiseSeparable(network, weightMap, *conv5_3->getOutput(0), 512, 512, 512, 1, 1, scale, 5, 2, false, "backbone.block_list.9");
    IElementWiseLayer* conv5_5 = depthWiseSeparable(network, weightMap, *conv5_4->getOutput(0), 512, 512, 512, 1, 1, scale, 5, 2, false, "backbone.block_list.10");
    IElementWiseLayer* conv5_6 = depthWiseSeparable(network, weightMap, *conv5_5->getOutput(0), 512, 1024, 512, 2, 1, scale, 5, 2,true, "backbone.block_list.11");

    IElementWiseLayer* conv6 = depthWiseSeparable(network, weightMap, *conv5_6->getOutput(0), 1024, 1024, 1024, 1, 2, scale, 5, 2, true, "backbone.block_list.12");

    // w/32, h/8
    IPoolingLayer* pool = network->addPoolingNd(*conv6->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool);
    pool->setStrideNd(DimsHW{2, 2});

    // neck encode with svtr  (1, 512, 1, -1)
    IElementWiseLayer* ew1 = convBnSwish(network, weightMap, *pool->getOutput(0), 64, 3, 1, 1, 1, 1, "neck.encoder.conv1");
    IElementWiseLayer* ew2 = convBnSwish(network, weightMap, *ew1->getOutput(0), 120, 1, 1, 1, 0, 1, "neck.encoder.conv2");

    // (1, 120, 1, -1)
    IShuffleLayer* sf1 = network->addShuffle(*ew2->getOutput(0));
    sf1->setReshapeDimensions(Dims{5, {1, 120, -1, 1, 1}});
    sf1->setSecondTranspose(Permutation{0, 2, 1, 3, 4});

    // (1, -1, 120, 1, 1)
    IElementWiseLayer* block1 = block(network, weightMap, *sf1->getOutput(0), 120, 8, 2, "neck.encoder.svtr_block.0");
    IElementWiseLayer* block2 = block(network, weightMap, *block1->getOutput(0), 120, 8, 2, "neck.encoder.svtr_block.1");
    IScaleLayer* norm = addLayerNorm(network, weightMap, *block2->getOutput(0), "neck.encoder.norm", 1e-6);

    IShuffleLayer* sf2 = network->addShuffle(*norm->getOutput(0));
    sf2->setReshapeDimensions(Dims4{1, -1, 1, 120});
    sf2->setSecondTranspose(Permutation{0, 3, 2, 1});

    IElementWiseLayer* ew3 = convBnSwish(network, weightMap, *sf2->getOutput(0), 512, 1, 1, 1, 0, 1, "neck.encoder.conv3");
    ITensor* input_tensor[] = {pool->getOutput(0), ew3->getOutput(0)};
    IConcatenationLayer* cat = network->addConcatenation(input_tensor, 2);
    IElementWiseLayer* ew4 = convBnSwish(network, weightMap, *cat->getOutput(0), 64, 3, 1, 1, 1, 1, "neck.encoder.conv4");
    IElementWiseLayer* ew5 = convBnSwish(network, weightMap, *ew4->getOutput(0), 64, 1, 1, 1, 0, 1, "neck.encoder.conv1x1");

    IShuffleLayer* sf3 = network->addShuffle(*ew5->getOutput(0));
    sf3->setReshapeDimensions(Dims{5, {1, 64, -1, 1, 1}});
    sf3->setSecondTranspose(Permutation{0, 2, 1, 3, 4});

    // head (1, -1, 64, 1, 1) => (1, -1, 6625, 1, 1)
    IFullyConnectedLayer* predicts = network->addFullyConnected(*sf3->getOutput(0), NUM_CLASS, weightMap["head.fc.weight"], weightMap["head.fc.bias"]);
    ISoftMaxLayer* result = network->addSoftMax(*predicts->getOutput(0));
    result->setAxes(4);

    // (1, -1, 6625, 1, 1) -> (1, -1, 1, 1, 1)
//    ITopKLayer* top = network->addTopK(*result->getOutput(0), TopKOperation::kMAX, 1, 4);
//    top->getOutput(1)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*top->getOutput(1));

    result->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*result->getOutput(0));

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1, 3, INPUT_H, 32));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(1, 3, INPUT_H, 480));
    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(1, 3, INPUT_H, 2048));
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
    std::string wts_path = "../../../weights/ch_ptocr_v3_rec_infer.wts";

    APIToModel(&modelStream, wts_path);
    assert(modelStream != nullptr);
    std::ofstream p("../../../weights/ch_ptocr_v3_rec_infer.engine", std::ios::binary);
    if(!p){
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    modelStream->destroy();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
