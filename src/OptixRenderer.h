//
// Created by Artyom on 12/3/2024.
//

#ifndef OPTIX_PYTORCH_EXTENSION_OPTIXRENDERER_H
#define OPTIX_PYTORCH_EXTENSION_OPTIXRENDERER_H
#define NOMINMAX
#define _USE_MATH_DEFINES
#include <cmath>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip> // for std::hex and std::dec
#include <fstream>
#include <vector>
#include <cuda.h>
#include "../sutil/helper_math.h"
#include <torch/extension.h>
#include "../sutil/Exception.h"
#include "../sutil/Preprocessor.h"
#include "optixDynamicGeometry.h"
#include <optix_stack_size.h>
#include "vertices.h"
#include "cube.h"
#include "TextureObject.h"
template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;
class OptixRenderer;
//TODO: change to Geometry + GeometryInstance
struct Geometry{
    torch::Tensor& vertices;
    torch::Tensor* indices = nullptr;
    torch::Tensor* texCoords = nullptr;
    OptixTraversableHandle optixTraversableHandle;
    CUdeviceptr d_gas_output_buffer;
    size_t      gas_output_buffer_size = 0;
    OptixTraversableHandle gas_handle;
    OptixRenderer& renderer;
    OptixInstance instance;
    OptixBuildInput build_input;
    OptixAccelEmitDesc emitProperty = {};
    HitGroupData material = {make_float3(1.0f, 1.0f, 1.0f), 1.4f, false, make_float3(0.0f, 0.0f, 0.0f)};
    //Geometry(torch::Tensor& vertices, torch::Tensor& indices): vertices(vertices), indices(indices){}
    Geometry(OptixRenderer& renderer, torch::Tensor& vertices, torch::Tensor* indices = nullptr):
            renderer(renderer), vertices(vertices), indices(indices)
    {}

    void setEmission(float r, float g, float b){
        material.emission = make_float3(r, g, b);
    }

    void setMaterialColor(float r, float g, float b){
        material.color = make_float3(r, g, b);
    }

    void setGlass(bool is_glass){
        material.is_glass = is_glass;
    }

    void setTexture(TextureObject& texture){
        material.texture = texture.get();
    }

    Geometry copy();
    Geometry compress();

};

class OptixRenderer {
public:
    OptixRenderer(int width, int height);
    ~OptixRenderer();

    void render(torch::Tensor& output_tensor, const std::vector<float>& camera_params, bool updated);
    //void set_geometry(const torch::Tensor& vertices, const torch::Tensor& indices);

    void createContext();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();
    void initLaunchParams();
    void buildMeshAccel();
    void updateMeshAccel();
    void launchSubframe(torch::Tensor& output_tensor);
    void updateState( torch::Tensor& output_tensor);
    void launchGenerateAnimatedVertices(AnimationMode animation_mode, CUdeviceptr& d_temp_vertices);
    Geometry createVertexGeometry(torch::Tensor& vertices,
                                                 torch::Tensor* indices,
                                                 torch::Tensor* texCoords,
                                                 torch::Tensor* tangents,
                                                 torch::Tensor* bitangents,
                                                 torch::Tensor* vertex_normals,
                                                 TextureObject* texture,
                                                 TextureObject* normals,
                                                 TextureObject* metallic_roughness,
                                                 TextureObject* emission_texture);
    uint32_t addGeometryInstance(Geometry& geometry);
    torch::Tensor getTransformForInstance(uint32_t id);
    OptixDeviceContext getContext() const{return context;}
    void buidIAS();
    //OptixTraversableHandle buildGASFromGeometry(Geometry& geometry);

    struct SBTData {
        CUdeviceptr d_sbt_records;       // Device pointer for SBT records
        size_t record_size;              // Size of one SBT record
        size_t capacity;                 // Total capacity (number of records)
        size_t count;                    // Current number of records
    };
private:


    OptixDeviceContext context = 0;

    size_t                         temp_buffer_size = 0;
    CUdeviceptr                    d_temp_buffer = 0;
    CUdeviceptr                    d_temp_vertices = 0;
    CUdeviceptr                    d_instances = 0;

    unsigned int                   triangle_flags = OPTIX_GEOMETRY_FLAG_NONE;//OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT

    OptixBuildInput                ias_instance_input = {};
    OptixBuildInput                triangle_input = {};

    OptixTraversableHandle         ias_handle;
    //OptixTraversableHandle         static_gas_handle;
    //OptixTraversableHandle         deforming_gas_handle;
    //OptixTraversableHandle         exploding_gas_handle;

    CUdeviceptr                    d_ias_output_buffer = 0;
    //CUdeviceptr                    d_static_gas_output_buffer;
    //CUdeviceptr                    d_deforming_gas_output_buffer;
    //CUdeviceptr                    d_exploding_gas_output_buffer;

    size_t                         ias_output_buffer_size = 0;
    //size_t                         static_gas_output_buffer_size = 0;
    //size_t                         deforming_gas_output_buffer_size = 0;
    //size_t                         exploding_gas_output_buffer_size = 0;

    std::vector<OptixInstance> instances;

    OptixModule                    ptx_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;

    OptixProgramGroup              any_hit_prog_group = 0;
    OptixProgramGroup              raygen_prog_group = 0;
    OptixProgramGroup              miss_group = 0;
    OptixProgramGroup              hit_group = 0;

    CUstream                       stream = 0;
    Params                         params;
    Params*                        d_params;

    float                          time = 0.f;
    float                          last_exploding_sphere_rebuild_time = 0.f;

    SBTData                        sbt_data;

    OptixShaderBindingTable        sbt = {};

    OptixAccelBufferSizes gas_buffer_sizes = {};

    void config_triangle_input(uint32_t numVertices, CUdeviceptr& d_temp_vertices);

    Geometry buidSphereGeometry();
    //void addNewGeometryInstance(std::vector<OptixInstance>& instances, OptixTraversableHandle gas_handle);
    void initSBT(size_t initial_capacity);
    void addGeometryToSBT(Geometry& geometry);
    void updateSBT();
};


#endif//OPTIX_PYTORCH_EXTENSION_OPTIXRENDERER_H
