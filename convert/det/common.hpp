#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include "NvInfer.h"
#include "cuda_utils.h"

using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val;
        val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// batch norm
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    scale_1->setName(lname.c_str());

    return scale_1;
}

// Conv
ILayer* convBnAct(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s, int p, int g , std::string act, std::string lname){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, c2, DimsHW{k, k}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname+".conv").c_str());
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname+".bn", 1e-5);
    if(act == "relu"){
        IActivationLayer* relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        return relu;
    }else if(act == "hard_swish"){
        IActivationLayer* hsig1 = network->addActivation(*bn1->getOutput(0), ActivationType::kHARD_SIGMOID);
        assert(hsig1);
        hsig1->setAlpha(1.0/6.0);
        hsig1->setBeta(0.5);

        IElementWiseLayer* ew1 = network->addElementWise(*bn1->getOutput(0), *hsig1->getOutput(0), ElementWiseOperation::kPROD);
        assert(ew1);
        return ew1;
    } else if(act == "swish"){
        IActivationLayer* sig1 = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
        IElementWiseLayer* ew1 = network->addElementWise(*bn1->getOutput(0), *sig1->getOutput(0), ElementWiseOperation::kPROD);
        assert(ew1);
        return ew1;
    }
    return bn1;
}

// SE
IElementWiseLayer* seLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, std::string lname){
    // global avg pooling
    IReduceLayer* reduce1 = network->addReduce(input, ReduceOperation::kAVG, 12, true);

    IConvolutionLayer* conv1 = network->addConvolutionNd(*reduce1->getOutput(0), c2/4, DimsHW{1,1}, weightMap[lname+".conv1.weight"], weightMap[lname+".conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setName((lname+".conv1").c_str());

    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), c2, DimsHW{1, 1}, weightMap[lname+".conv2.weight"], weightMap[lname+".conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setName((lname+".conv2").c_str());

    IActivationLayer* hsig1 = network->addActivation(*conv2->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig1);
    hsig1->setAlpha(1.2/6.0);
    hsig1->setBeta(0.5);

    IElementWiseLayer* ew1 = network->addElementWise(input, *hsig1->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew1);
    return ew1;
}

// ResidualUnit
ILayer* residualUnit(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int c3, int k,  int s, bool use_se, bool shortcut, std::string act, std::string lname){
    ILayer* expand_conv = convBnAct(network, weightMap, input, c2, 1, 1, 0, 1, act, lname+".expand_conv");
    ILayer* bottleneck_conv = convBnAct(network, weightMap, *expand_conv->getOutput(0), c2, k, s, int((k-1)/2), c2, act, lname+".bottleneck_conv");
    ILayer* linear_conv = nullptr;
    if(use_se){
        IElementWiseLayer* mid = seLayer(network, weightMap, *bottleneck_conv->getOutput(0), c2, lname+".mid");
        linear_conv = convBnAct(network, weightMap, *mid->getOutput(0), c3, 1, 1, 0, 1, "none", lname+".linear_conv");
    }else{
        linear_conv = convBnAct(network, weightMap, *bottleneck_conv->getOutput(0), c3, 1, 1, 0, 1, "none", lname+".linear_conv");
    }
    assert(linear_conv);

    if(shortcut){
        IElementWiseLayer* ew1 = network->addElementWise(input, *linear_conv->getOutput(0), ElementWiseOperation::kSUM);
        return ew1;
    }
    return linear_conv;

}

// RSE
IElementWiseLayer* rseLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, bool shortcut, std::string lname){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* in_conv = network->addConvolutionNd(input, c2, DimsHW{k, k}, weightMap[lname+".in_conv.weight"], emptywts);
    assert(in_conv);
    in_conv->setName((lname+".in_conv").c_str());
    in_conv->setPaddingNd(DimsHW{int(k/2), int(k/2)});

    if(shortcut){
        IElementWiseLayer* se_block = seLayer(network, weightMap, *in_conv->getOutput(0), c2, lname+".se_block");
        IElementWiseLayer* ew1 = network->addElementWise(*in_conv->getOutput(0), *se_block->getOutput(0), ElementWiseOperation::kSUM);
        return ew1;
    }else{
        IElementWiseLayer* se_block = seLayer(network, weightMap, *in_conv->getOutput(0), c2, lname+".se_block");
        return se_block;
    }

}

#endif
