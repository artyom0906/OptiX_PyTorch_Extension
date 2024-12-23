// renderer.h

#ifndef RENDERER_H
#define RENDERER_H

#define NOMINMAX
#define _USE_MATH_DEFINES
#include <cmath>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "../sutil/helper_math.h"
#include <torch/extension.h>
#include "../sutil/Exception.h"

struct Params {
    float3* image;
    int width;
    int height;
    float3 eye;
    float3 U;
    float3 V;
    float3 W;
    OptixTraversableHandle handle;
};

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    void render(torch::Tensor& output_tensor, const std::vector<float>& camera_params);
    void set_geometry(const torch::Tensor& vertices, const torch::Tensor& indices);
    void buildSBT();
    void buildAccelerationStructure();
private:
    // OptiX context and pipeline
    OptixDeviceContext context = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    OptixShaderBindingTable sbt = {};

    // Acceleration structure
    OptixTraversableHandle gas_handle = 0; // Initialize to zero
    CUdeviceptr d_gas_output_buffer = 0;

    // SBT records
    CUdeviceptr d_raygen_record = 0;
    CUdeviceptr d_miss_record = 0;
    CUdeviceptr d_hitgroup_record = 0;

    // Geometry data
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;

    //
    unsigned int num_vertices = 0;
    unsigned int num_indices = 0;

    // Parameters buffer
    CUdeviceptr d_params = 0;

    // Other members
    int width;
    int height;

    // Pipeline compile options
    OptixPipelineCompileOptions pipeline_compile_options = {};

    // Initialization methods
    void initOptix();
    void createContext();
    void buildModule();
    void createProgramGroups();
    void createPipeline();
};

#endif // RENDERER_H
