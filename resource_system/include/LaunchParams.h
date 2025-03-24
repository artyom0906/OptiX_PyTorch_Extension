#pragma once

#include <vector_types.h>
#include <optix.h>
#include "../src/kernels/material_types.h"

namespace optix_renderer {

// Launch parameters structure shared between host and device
// IMPORTANT: This must exactly match the Params struct in shader_minimal.cu
struct LaunchParams {
    float3* image;          // Raw pointer to output buffer
    float3 camera_pos;
    float3 camera_u;
    float3 camera_v;
    float3 camera_w;
    OptixTraversableHandle traversable;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_pixel;
    unsigned int max_depth;
    CUdeviceptr output_buffer;  // Additional field for direct device ptr access
};

// SBT records for the different program types
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RayGenSbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissSbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Payload can be added here if needed
};

// Hit group data - will be the payload for hit group records
struct HitGroupData {
    SimpleMaterial material;
    
    // Geometry data for this hit group
    CUdeviceptr vertices;
    CUdeviceptr indices;
    CUdeviceptr normals;
    CUdeviceptr texcoords;
    CUdeviceptr tangents;
    CUdeviceptr bitangents;
    
    // Texture data using CUtexObject for driver API compatibility
    CUtexObject albedo_texture;
    CUtexObject normal_texture;
    CUtexObject metallic_roughness_texture;
    CUtexObject emission_texture;
    CUtexObject specular_texture;
    CUtexObject specular_tint_texture;
    CUtexObject sheen_texture;
    CUtexObject clearcoat_texture;
    
    // Additional settings
    bool has_normals;
    bool has_texcoords;
    bool has_tangents;
    bool has_bitangents;
    
    // Legacy compatibility fields
    float3 albedo;        // Direct color access for basic shading
    float3 emission;      // Direct emission color
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupSbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    HitGroupData data;
};

} // namespace optix_renderer