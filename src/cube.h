//
// Created by Artyom on 12/3/2024.
//

#ifndef OPTIX_PYTORCH_EXTENSION_CUBE_H
#define OPTIX_PYTORCH_EXTENSION_CUBE_H
#include <optix.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// Declare the vertices and indices as extern
extern std::vector<float3> cube_vertices;
extern std::vector<uint3> cube_indices;

// Function to create a tensor from std::vector<float3>
torch::Tensor create_vertices_tensor(const std::vector<float3>& cube_vertices);

// Function to create a tensor from std::vector<uint3>
torch::Tensor create_indices_tensor(const std::vector<uint3>& cube_indices);



#endif//OPTIX_PYTORCH_EXTENSION_CUBE_H
