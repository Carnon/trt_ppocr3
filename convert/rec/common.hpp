#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include "NvInfer.h"
#include "cuda_utils.h"

static const float SCALING_ONE = 1.0;
static const float SHIFT_ZERO = 0.0;
static const float POWER_TWO = 2.0;
static const float EPS = 0.00001;

using namespace nvinfer1;


void debug_print(ITensor *input_tensor,std::string head){
    std::cout << head<< " : ";

    for (int i = 0; i < input_tensor->getDimensions().nbDims; i++)
    {
        std::cout << input_tensor->getDimensions().d[i] << " ";
    }
    std::cout<<std::endl;

}


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

// layer norm
IScaleLayer* addLayerNorm(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps){
    // (1, -1, 120, 1, 1)
    int len = int(weightMap[lname + ".bias"].count);

    IReduceLayer* mean = network->addReduce(input, ReduceOperation::kAVG, 4, true);
    assert(mean);

    IElementWiseLayer* sub_mean = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(sub_mean);
    // implement pow2 with scale
    Weights scale{ DataType::kFLOAT, &SCALING_ONE, 1 };
    Weights shift{ DataType::kFLOAT, &SHIFT_ZERO, 1 };
    Weights power{ DataType::kFLOAT, &POWER_TWO, 1 };
    IScaleLayer* pow2 = network->addScale(*sub_mean->getOutput(0), ScaleMode::kUNIFORM, shift, scale, power);
    assert(pow2);

    IReduceLayer* pow_mean = network->addReduce(*pow2->getOutput(0), ReduceOperation::kAVG, 4, true);
    assert(pow_mean);

    IConstantLayer* eps_constant = network->addConstant(Dims{5, {1, 1, 1, 1, 1}}, Weights{DataType::kFLOAT, &EPS, 1});
    assert(eps);

    IElementWiseLayer* add_eps = network->addElementWise(*pow_mean->getOutput(0), *eps_constant->getOutput(0), ElementWiseOperation::kSUM);

    IUnaryLayer* sqrt = network->addUnary(*add_eps->getOutput(0), UnaryOperation::kSQRT);
    assert(sqrt);

    IElementWiseLayer* div = network->addElementWise(*sub_mean->getOutput(0), *sqrt->getOutput(0), ElementWiseOperation::kDIV);
    assert(div);

    auto *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));

    for (int i = 0; i < len; i++) pval[i] = 1.0;

    Weights norm1_power{ DataType::kFLOAT, pval, len};
    weightMap[lname + ".power"] = norm1_power;

    IScaleLayer* scale_1 = network->addScaleNd(*div->getOutput(0), ScaleMode::kCHANNEL, weightMap[lname+".bias"], weightMap[lname+".weight"], norm1_power, 2);
    assert(scale_1);
    scale_1->setName(lname.c_str());
    return scale_1;
}

// Conv
IElementWiseLayer* convBnHardSwish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s1, int s2, int p, int g, std::string lname){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, c2, DimsHW{k, k}, weightMap[lname + "._conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname+".conv").c_str());
    conv1->setStrideNd(DimsHW{s1, s2});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname+"._batch_norm", 1e-5);

    IActivationLayer* hsig1 = network->addActivation(*bn1->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig1);
    hsig1->setAlpha(1.0/6.0);
    hsig1->setBeta(0.5);

    IElementWiseLayer* ew1 = network->addElementWise(*bn1->getOutput(0), *hsig1->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew1);
    return ew1;
}

// Conv
IElementWiseLayer* convBnSwish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s1, int s2, int p, int g, std::string lname){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, c2, DimsHW{k, k}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname+".conv").c_str());
    conv1->setStrideNd(DimsHW{s1, s2});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname+".norm", 1e-5);

    // silu = x * sigmoid(x)
    IActivationLayer* sig1 = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig1);
    IElementWiseLayer* ew1 = network->addElementWise(*bn1->getOutput(0), *sig1->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew1);
    return ew1;
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
    hsig1->setAlpha(1.0/6.0);
    hsig1->setBeta(0.5);

    IElementWiseLayer* ew1 = network->addElementWise(input, *hsig1->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew1);
    return ew1;
}


// DepthwiseSeparable
IElementWiseLayer* depthWiseSeparable(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int c3, int g, int s1, int s2, float scale, int dw, int p, bool useSe, std::string lname=""){
    int _c2 = int(float(c2)*scale);
    int _c3 = int(float(c3)*scale);
    int _g = int(float(g)*scale);

    IElementWiseLayer* ew1 = convBnHardSwish(network, weightMap, input, _c2, dw, s1, s2, p, _g, lname+"._depthwise_conv");
    if(useSe){
        IElementWiseLayer* ew2 = seLayer(network, weightMap, *ew1->getOutput(0), _c2, lname+"._se");
        IElementWiseLayer* ew3 = convBnHardSwish(network, weightMap, *ew2->getOutput(0), _c3, 1, 1, 1, 0, 1, lname+"._pointwise_conv");
        return ew3;
    }else{
        IElementWiseLayer* ew3 = convBnHardSwish(network, weightMap, *ew1->getOutput(0), _c3, 1, 1, 1, 0, 1, lname+"._pointwise_conv");
        return ew3;
    }
}

// mlp
IFullyConnectedLayer* mlp(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int c3, std::string lname){
    // (1, -1, 120, 1, 1)
    IFullyConnectedLayer* fc1 = network->addFullyConnected(input, c2, weightMap[lname+".fc1.weight"], weightMap[lname+".fc1.bias"]);
    fc1->setName((lname+".fc1").c_str());
    assert(fc1);

    IActivationLayer* act = network->addActivation(*fc1->getOutput(0), ActivationType::kSIGMOID);
    IElementWiseLayer* act_ = network->addElementWise(*fc1->getOutput(0), *act->getOutput(0), ElementWiseOperation::kPROD);

    IFullyConnectedLayer* fc2 = network->addFullyConnected(*act_->getOutput(0), c3, weightMap[lname+".fc2.weight"], weightMap[lname+".fc2.bias"]);
    assert(fc2);
    return fc2;
}

// (1, -1, 120, 1, 1) -> (1, -1, 120, 1, 1)
IFullyConnectedLayer* attention(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int dim, int heads, std::string lname){
    int len = 1;
    auto *scale = new float[len];
    for(int i=0; i< len; i++){
        scale[i] = 1.f / float(sqrt(dim/heads));
    }
    Weights scale_w{DataType::kFLOAT, nullptr, len};
    scale_w.values = scale;
    IConstantLayer* q_scale = network->addConstant(Dims{5, {1,1, 1, 1, 1}}, scale_w);

    // (1, -1, 120, 1, 1) -> (1, -1, 360, 1, 1)
    IFullyConnectedLayer* linear1 = network->addFullyConnected(input, dim*3, weightMap[lname+".qkv.weight"], weightMap[lname+".qkv.bias"]);

    // (1, -1, 360, 1, 1) -> (1, -1, 3, 8, 15) -> (1, 3, 8, -1, 15)
    IShuffleLayer* sf1 = network->addShuffle(*linear1->getOutput(0));
    sf1->setReshapeDimensions(Dims{5, {1, -1, 3, 8, 15}});
    sf1->setSecondTranspose(Permutation{0, 2, 3, 1, 4});

    IReduceLayer* reduce = network->addReduce(*sf1->getOutput(0), ReduceOperation::kAVG, 2, true);
    IShapeLayer* shape = network->addShape(*reduce->getOutput(0));
    // (3, 1, 8, -1, 15)
    ISliceLayer* q = network->addSlice(*sf1->getOutput(0),Dims{5, {0, 0, 0, 0, 0}},Dims{5, {1, 1, 8, 60, 15}},Dims{5, {1,1, 1, 1, 1}});
    q->setInput(2, *shape->getOutput(0));
    assert(q);
    ISliceLayer* k = network->addSlice(*sf1->getOutput(0), Dims{5, {0, 1, 0, 0, 0}}, Dims{5, {1,1, 8, 60, 15}}, Dims{5, {1,1, 1, 1, 1}});
    k->setInput(2, *shape->getOutput(0));
    ISliceLayer* v = network->addSlice(*sf1->getOutput(0), Dims{5, {0, 2, 0, 0, 0}}, Dims{5, {1,1, 8, 60, 15}}, Dims{5, {1,1, 1, 1, 1}});
    v->setInput(2, *shape->getOutput(0));

    // (1, 1, 8, -1, 15)
    IElementWiseLayer* qs = network->addElementWise(*q->getOutput(0), *q_scale->getOutput(0), ElementWiseOperation::kPROD);
    IShuffleLayer* qs_ = network->addShuffle(*qs->getOutput(0));
    qs_->setReshapeDimensions(Dims4{1, heads, -1, dim/heads});

    // (1, 1, 8 , -1, 15) * (1, 1, 8, -1, 15) => (1, 8, -1, -1) ???
    IMatrixMultiplyLayer* atten = network->addMatrixMultiply(*qs->getOutput(0), MatrixOperation::kNONE, *k->getOutput(0), MatrixOperation::kTRANSPOSE);
    ISoftMaxLayer* atten_softmax = network->addSoftMax(*atten->getOutput(0));
    atten_softmax->setAxes(16);

    // (1, 1, 8, -1, -1) * (1, 1, 8, -1, 15) => (1, 1, 8, -1, 15)
    IMatrixMultiplyLayer*  attn_product_v = network->addMatrixMultiply(*atten_softmax->getOutput(0), MatrixOperation::kNONE, *v->getOutput(0), MatrixOperation::kNONE);
    assert(attn_product_v);

    // (1, 1, 8, -1, 15) => (1, -1, 120, 1, 1)
    IShuffleLayer* sf4 = network->addShuffle(*attn_product_v->getOutput(0));
    sf4->setFirstTranspose(Permutation{1, 0, 3, 2, 4});
    sf4->setReshapeDimensions(Dims{ 5, {1,  -1, dim, 1, 1}});

    IFullyConnectedLayer* proj = network->addFullyConnected(*sf4->getOutput(0), dim, weightMap[lname+".proj.weight"], weightMap[lname+".proj.bias"]);
    assert(proj);
    return proj;
}


IElementWiseLayer* block(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int dim, int heads, float ratio, std::string lname){
    // (1, -1, 120, 1)
    auto ln1 = addLayerNorm(network, weightMap, input, lname+".norm1", 1e-5);
    IFullyConnectedLayer* mixer = attention(network, weightMap, *ln1->getOutput(0), dim, heads, lname+".mixer");
    IElementWiseLayer* ew1 = network->addElementWise(input, *mixer->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew1);
    auto ln2 = addLayerNorm(network, weightMap, *ew1->getOutput(0), lname+".norm2", 1e-5);
    IFullyConnectedLayer* fc = mlp(network, weightMap, *ln2->getOutput(0), int(dim*ratio), dim, lname+".mlp");
    assert(fc);
    IElementWiseLayer* ew2 = network->addElementWise(*ew1->getOutput(0), *fc->getOutput(0), ElementWiseOperation::kSUM);
    return ew2;
}

#endif
