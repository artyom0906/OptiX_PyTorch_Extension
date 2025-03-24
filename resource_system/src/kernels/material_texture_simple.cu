#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>  // For CUtexObject
#include <vector_types.h>
#include <vector_functions.h>
#include "material_types.h"  // Include the enhanced material types

using namespace optix_renderer;

// Vector math helpers
__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float3 normalize(const float3& v) {
    float invLen = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return v * invLen;
}

__device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// The hit group data structure (simplified)
struct HitGroupData {
    // Material data
    SimpleMaterial material;
    
    // Geometry data
    CUdeviceptr vertices;
    CUdeviceptr indices;
    CUdeviceptr texcoords;
    
    // Texture objects
    CUtexObject albedo_texture;
    
    // Flags for geometry features
    bool has_texcoords;
    
    // Legacy compatibility
    float3 albedo;
};

// Launch parameters structure must match the one in LaunchParams.h
struct Params {
    float3* image;
    float3 camera_pos;
    float3 camera_u;
    float3 camera_v;
    float3 camera_w;
    OptixTraversableHandle traversable;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_pixel;
    unsigned int max_depth;
    CUdeviceptr output_buffer;
};

extern "C" {
    __constant__ Params params;
}

// Compute triangle normal
__device__ float3 computeNormal(const HitGroupData* data, const float2& barycentrics) {
    // Get the triangle index
    const int primitiveIndex = optixGetPrimitiveIndex();
    
    // Access vertex data
    if (!data->vertices) {
        // Default normal if no geometry data
        return make_float3(0.0f, 1.0f, 0.0f);
    }
    
    // Get triangle indices
    int3 indices;
    if (data->indices) {
        indices = *((int3*)(data->indices) + primitiveIndex);
    } else {
        // Direct indexing if no index buffer
        indices.x = 3 * primitiveIndex;
        indices.y = 3 * primitiveIndex + 1;
        indices.z = 3 * primitiveIndex + 2;
    }
    
    // Get vertex positions
    float3* vertices = (float3*)(data->vertices);
    float3 v0 = vertices[indices.x];
    float3 v1 = vertices[indices.y];
    float3 v2 = vertices[indices.z];
    
    // Compute geometric normal
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 normal = cross(e1, e2);
    
    return normalize(normal);
}

// Get texture coordinates at hit point - simplified version
__device__ float2 getTexCoords(const HitGroupData* data, const float2& barycentrics) {
    if (!data->has_texcoords || !data->texcoords) {
        return make_float2(0.5f, 0.5f);  // Default UV in the middle of texture
    }
    
    // Get primitive index and indices
    const int primitiveIndex = optixGetPrimitiveIndex();
    int3 indices;
    if (data->indices) {
        indices = *((int3*)(data->indices) + primitiveIndex);
    } else {
        indices.x = 3 * primitiveIndex;
        indices.y = 3 * primitiveIndex + 1;
        indices.z = 3 * primitiveIndex + 2;
    }
    
    // Get texture coordinates
    float2* texcoords = (float2*)(data->texcoords);
    float2 tc0 = texcoords[indices.x];
    float2 tc1 = texcoords[indices.y];
    float2 tc2 = texcoords[indices.z];
    
    // Interpolate using barycentric coordinates
    float w0 = 1.0f - barycentrics.x - barycentrics.y;
    float w1 = barycentrics.x;
    float w2 = barycentrics.y;
    
    // Perform interpolation component-wise
    float2 result;
    result.x = w0 * tc0.x + w1 * tc1.x + w2 * tc2.x;
    result.y = w0 * tc0.y + w1 * tc1.y + w2 * tc2.y;
    
    return result;
}

// Sample texture if it exists, otherwise return default color
__device__ float4 sampleTexture(CUtexObject tex, float2 uv) {
    if (tex) {
        return tex2D<float4>(tex, uv.x, uv.y);
    }
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);  // Magenta if texture missing
}

// Ray generation program with simplified texture rendering
extern "C" __global__ void __raygen__simple() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    // Calculate buffer index
    unsigned int x = idx.x;
    unsigned int y = idx.y;
    
    // Use dimensions from params if valid
    unsigned int width = dim.x;
    unsigned int height = dim.y;
    if (params.width > 0) width = params.width;
    if (params.height > 0) height = params.height;
    
    // Ensure we're within bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate normalized screen coordinates [-1,1]
    const float2 screen = make_float2(
        (float)x / width * 2.0f - 1.0f,
        (float)y / height * 2.0f - 1.0f
    );
    
    // Calculate ray parameters
    const float3 origin = params.camera_pos;
    
    // Calculate ray direction using camera basis vectors
    float3 direction = normalize(
        params.camera_u * screen.x + 
        params.camera_v * screen.y + 
        params.camera_w
    );
    
    // Trace the ray
    unsigned int p0 = 0;  // Red
    unsigned int p1 = 0;  // Green
    unsigned int p2 = 0;  // Blue
    
    // Default fallback color (dark blue background)
    float3 backgroundColor = make_float3(0.0f, 0.0f, 0.2f);
    
    if (params.traversable) {
        optixTrace(
            params.traversable,
            origin,
            direction,
            0.0f,                  // Min distance
            1e16f,                 // Max distance
            0.0f,                  // Ray time
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,                     // SBT offset
            1,                     // SBT stride
            0,                     // Miss SBT index
            p0, p1, p2             // Payload (RGB values)
        );
        
        // Convert payload to float RGB values
        backgroundColor.x = __uint_as_float(p0);
        backgroundColor.y = __uint_as_float(p1);
        backgroundColor.z = __uint_as_float(p2);
    }
    
    // Output to buffer
    float3* output = (float3*)params.image;
    if (params.output_buffer != 0) {
        unsigned int idx = y * width + x;
        if (x < width && y < height) {
            output[idx] = backgroundColor;
        }
    }
}

// Miss shader that returns a grid pattern
extern "C" __global__ void __miss__grid() {
    const float3 rayDir = normalize(optixGetWorldRayDirection());
    
    // Simple grid pattern
    float3 baseColor = make_float3(0.2f, 0.2f, 0.3f); // Dark blue-gray
    
    // Add virtual horizon grid
    float3 horizonColor = make_float3(0.6f, 0.6f, 0.8f); // Light blue-gray
    
    // Grid pattern
    float grid = 0.0f;
    
    // Calculate direction on xz plane
    float2 dir = make_float2(rayDir.x, rayDir.z);
    float len = sqrtf(dir.x * dir.x + dir.y * dir.y);
    if (len > 0.0f) {
        dir.x /= len;
        dir.y /= len;
    }
    
    float y = rayDir.y;
    if (fabsf(y) < 0.1f) {
        // Add horizontal grid lines
        grid = 0.2f;
    }
    
    // Apply grid with properly typecasted operations
    float3 gridColor = make_float3(grid, grid, grid) * horizonColor;
    float3 color = baseColor + gridColor;
    
    // Set the payload to the computed color
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

// Ultra-simplified closest hit shader that just shows textures
extern "C" __global__ void __closesthit__texture() {
    // Get the SBT record data
    const HitGroupData* data = (const HitGroupData*)optixGetSbtDataPointer();
    
    // Get hit information
    const float2 barycentrics = optixGetTriangleBarycentrics();
    
    // Get texture coordinates
    float2 texCoords = getTexCoords(data, barycentrics);
    
    // Default color is the material albedo
    float3 color = data->material.albedo;
    
    // Check if we have an albedo texture
    //if (data->albedo_texture) {
    //    // We have a texture! Sample it
    //    float4 texColor = sampleTexture(data->albedo_texture, texCoords);
    //
    //    // Use the texture color but keep a bit of the material color visible at edges
    //    // for better visibility of texture coordinates
    //    float3 texColorRGB = make_float3(texColor.x, texColor.y, texColor.z);
    //    color = texColorRGB;
    //
    //    // Add UV coordinate visualization (red for u, green for v)
    //    float uvBorder = 0.02f; // Border size
    //    if (texCoords.x < uvBorder || texCoords.x > 1.0f - uvBorder ||
    //        texCoords.y < uvBorder || texCoords.y > 1.0f - uvBorder) {
    //        color = make_float3(texCoords.x, texCoords.y, 0.0f);
    //    }
    //} else {
        // If no texture, use simple shading with normal
        float3 normal = computeNormal(data, barycentrics);
        float ndotl = max(0.0f, normal.y * 0.5f + 0.5f); // Simple light from above
        
        // Mix original color with lighting
        color = color * ndotl;
        
        // Add UV coordinate visualization if we have texture coordinates
        if (data->has_texcoords) {
            float uvBorder = 0.02f; // Border size
            if (texCoords.x < uvBorder || texCoords.x > 1.0f - uvBorder || 
                texCoords.y < uvBorder || texCoords.y > 1.0f - uvBorder) {
                // Show UV coordinates along edges
                color = make_float3(texCoords.x, texCoords.y, 0.0f);
            }
        }
    //}
    
    // Set the payload values to our computed color
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}