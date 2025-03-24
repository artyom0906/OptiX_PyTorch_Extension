//
// Created by artem on 2/28/25.
//

#ifndef GEOMETRY_H
#define GEOMETRY_H
#include <optix.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda.h>
#include "optixDynamicGeometry.h"
#include "TextureObject.h"
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
#endif //GEOMETRY_H
