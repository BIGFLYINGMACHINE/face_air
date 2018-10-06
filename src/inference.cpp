//
// Created by zack on 18-10-6.
//
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "common.h"
#include "NvUtils.h"
#include "sample_functions.h"
#include "sampleMNIST.h"
using namespace nvuffparser;
using namespace nvinfer1;


std::vector< std::vector<float> > get_face_embeddings(int num_faces, bool building_engine)
{
    // this gLogger is static
    IBuilder* builder = createInferBuilder(gLogger);
    int maxBatchSize = 20;
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    INetworkDefinition* network = builder->createNetwork();
    auto parser = createUffParser();
    parser->registerInput("Input_0", DimsCHW(3, 160, 160), UffInputOrder::kNCHW);
    parser->registerOutput("face_embeddings");

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    IExecutionContext *context = engine->createExecutionContext();
    int inputIndex = engine->getBindingIndex("Input_0"),
            outputIndex = engine->getBindingIndex("face_embeddings");

}

