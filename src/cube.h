//
// Created by Artyom on 12/3/2024.
//

#ifndef OPTIX_PYTORCH_EXTENSION_CUBE_H
#define OPTIX_PYTORCH_EXTENSION_CUBE_H
#include <optix.h>
// Define the vertices of the cube
std::vector<float3> cube_vertices = {
        // Front face
        { -1.0f, -1.0f,  1.0f },
        {  1.0f, -1.0f,  1.0f },
        {  1.0f,  1.0f,  1.0f },
        { -1.0f,  1.0f,  1.0f },
        // Back face
        { -1.0f, -1.0f, -1.0f },
        {  1.0f, -1.0f, -1.0f },
        {  1.0f,  1.0f, -1.0f },
        { -1.0f,  1.0f, -1.0f },
};
// Define the indices for the cube (two triangles per face)
std::vector<uint3> cube_indices = {
        // Front face
        { 0, 1, 2 },
        { 0, 2, 3 },
        // Right face
        { 1, 5, 6 },
        { 1, 6, 2 },
        // Back face
        { 5, 4, 7 },
        { 5, 7, 6 },
        // Left face
        { 4, 0, 3 },
        { 4, 3, 7 },
        // Top face
        { 3, 2, 6 },
        { 3, 6, 7 },
        // Bottom face
        { 4, 5, 1 },
        { 4, 1, 0 },
};
// Function to create a tensor from std::vector<float3>
torch::Tensor create_vertices_tensor(const std::vector<float3> cube_vertices) {
    // Reinterpret the data pointer as a float pointer
    auto data_ptr = reinterpret_cast<const float*>(cube_vertices.data());

    // Create a tensor from the data pointer
    // Specify the size [num_vertices, 3] and the data type
    torch::Tensor vertices_tensor = torch::from_blob(
                                            const_cast<float*>(data_ptr),  // Remove constness (safe here because we're not modifying)
                                            { static_cast<int64_t>(cube_vertices.size()), 3 },
                                            torch::TensorOptions().dtype(torch::kFloat32)
                                                    ).clone(); // Clone to ensure the tensor owns its data
    vertices_tensor = vertices_tensor.to(torch::kCUDA);
    return vertices_tensor;
}

// Function to create a tensor from std::vector<uint3>
torch::Tensor create_indices_tensor(const std::vector<uint3> cube_indices) {
    // Reinterpret the data pointer as a uint32_t pointer
    auto data_ptr = reinterpret_cast<const uint32_t*>(cube_indices.data());

    // Create a tensor from the data pointer
    torch::Tensor indices_tensor = torch::from_blob(
                                           const_cast<uint32_t*>(data_ptr),  // Remove constness
                                           { static_cast<int64_t>(cube_indices.size()), 3 },
                                           torch::TensorOptions().dtype(torch::kUInt32)
                                                   ).clone(); // Clone to ensure the tensor owns its data
    indices_tensor = indices_tensor.to(torch::kCUDA);
    return indices_tensor;
}



#endif//OPTIX_PYTORCH_EXTENSION_CUBE_H
