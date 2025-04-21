#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "material_types.h"

using namespace optix_renderer;

// Vector math helper functions
__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(const float3& v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

__device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Hit group data structure
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

// Launch parameters
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

// Get texture coordinates at hit point
__device__ float2 getTexCoords(const HitGroupData* data, const float2& barycentrics) {
    if (!data->has_texcoords || !data->texcoords) {
        return make_float2(0.5f, 0.5f);
    }

    const int primitiveIndex = optixGetPrimitiveIndex();
    int3 indices;
    if (data->indices) {
        indices = *((int3*)(data->indices) + primitiveIndex);
    } else {
        indices.x = 3 * primitiveIndex;
        indices.y = 3 * primitiveIndex + 1;
        indices.z = 3 * primitiveIndex + 2;
    }

    float2* texcoords = (float2*)(data->texcoords);
    float2 tc0 = texcoords[indices.x];
    float2 tc1 = texcoords[indices.y];
    float2 tc2 = texcoords[indices.z];

    float w0 = 1.0f - barycentrics.x - barycentrics.y;
    float w1 = barycentrics.x;
    float w2 = barycentrics.y;

    float2 result;
    result.x = w0 * tc0.x + w1 * tc1.x + w2 * tc2.x;
    result.y = w0 * tc0.y + w1 * tc1.y + w2 * tc2.y;

    return result;
}

// Sample texture
__device__ float4 sampleTexture(CUtexObject tex, float2 uv) {
    return tex2D<float4>(tex, uv.x, uv.y);
}

extern "C" {
    __constant__ Params params;
}

// Ray generation shader
extern "C" __global__ void __raygen__renderFrame() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    unsigned int x = idx.x;
    unsigned int y = idx.y;

    unsigned int width = dim.x;
    unsigned int height = dim.y;

    if (params.width > 0) width = params.width;
    if (params.height > 0) height = params.height;

    if (x >= width || y >= height) {
        return;
    }

    float3 color = make_float3(0.1f, 0.1f, 0.1f);

    if (params.traversable) {
        const float3 ray_origin = params.camera_pos;

        // Flip y coordinate to make Y+ point up in camera space
        // Standard NDC has y from -1 (bottom) to 1 (top)
        // By flipping the y coordinate here, we make Y+ upward
        const float2 screen_pos = make_float2(
            (float)x / width * 2.0f - 1.0f,
            -((float)y / height * 2.0f - 1.0f) // Flip Y coordinate
        );

        float3 dir = make_float3(
            params.camera_u.x * screen_pos.x + params.camera_v.x * screen_pos.y + params.camera_w.x,
            params.camera_u.y * screen_pos.x + params.camera_v.y * screen_pos.y + params.camera_w.y,
            params.camera_u.z * screen_pos.x + params.camera_v.z * screen_pos.y + params.camera_w.z
        );

        // The Z- forward convention is handled in the camera basis calculation in setupLaunchParams
        // This is controlled by the camera "up" and "lookAt" vectors

        float length = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
        const float3 ray_direction = make_float3(dir.x/length, dir.y/length, dir.z/length);

        unsigned int p0 = 0;
        unsigned int p1 = 0;
        unsigned int p2 = 0;
        unsigned int sbtOffset = 0;

        optixTrace(
            params.traversable,
            ray_origin,
            ray_direction,
            0.0f,
            1e16f,
            0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            sbtOffset,
            1,
            0,
            p0, p1, p2
        );

        color.x = __uint_as_float(p0);
        color.y = __uint_as_float(p1);
        color.z = __uint_as_float(p2);
    }

    float3* output = (float3*)params.image;
    if (params.output_buffer != 0) {
        unsigned int idx = y * width + x;
        if (x < width && y < height) {
            output[idx] = color;
        }
    }
}

// Miss shader
extern "C" __global__ void __miss__environment() {
    float3 color = make_float3(0.1f, 0.1f, 0.2f);
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

// Get normal at hit point
__device__ float3 getNormal(const HitGroupData* data, const float2& barycentrics) {
    if (!data->has_normals || !data->normals) {
        // Return a default normal if normals are not available
        return make_float3(0.0f, 1.0f, 0.0f);
    }

    const int primitiveIndex = optixGetPrimitiveIndex();
    int3 indices;
    if (data->indices) {
        indices = *((int3*)(data->indices) + primitiveIndex);
    } else {
        indices.x = 3 * primitiveIndex;
        indices.y = 3 * primitiveIndex + 1;
        indices.z = 3 * primitiveIndex + 2;
    }

    float3* normals = (float3*)(data->normals);
    float3 n0 = normals[indices.x];
    float3 n1 = normals[indices.y];
    float3 n2 = normals[indices.z];

    float w0 = 1.0f - barycentrics.x - barycentrics.y;
    float w1 = barycentrics.x;
    float w2 = barycentrics.y;

    float3 result = make_float3(
        w0 * n0.x + w1 * n1.x + w2 * n2.x,
        w0 * n0.y + w1 * n1.y + w2 * n2.y,
        w0 * n0.z + w1 * n1.z + w2 * n2.z
    );

    return normalize(result);
}

// Simple lighting calculation
__device__ float3 calculateLighting(
    const float3& position,
    const float3& normal,
    const float3& view_dir,
    const float3& albedo,
    float metallic,
    float roughness
) {
    // Fixed light position for simplicity
    float3 light_dir = normalize(make_float3(1.0f, 1.0f, 1.0f));
    float3 light_color = make_float3(1.0f, 1.0f, 1.0f);
    float light_intensity = 1.5f;
    
    // Ambient lighting
    float ambient_strength = 0.2f;
    float3 ambient = ambient_strength * albedo;
    
    // Diffuse lighting (Lambert model)
    float diff = max(dot(normal, light_dir), 0.0f);
    float3 diffuse = diff * light_color * albedo;
    
    // Specular lighting (Blinn-Phong approximation)
    float3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0f), 32.0f); 
    float specular_strength = 0.5f * (1.0f - roughness);
    float3 specular = specular_strength * spec * light_color;
    
    // Combine components
    float3 result = ambient + (diffuse + specular) * light_intensity;
    
    // Ensure values are in valid range
    result.x = min(result.x, 1.0f);
    result.y = min(result.y, 1.0f);
    result.z = min(result.z, 1.0f);
    
    return result;
}

// Closest hit shader
extern "C" __global__ void __closesthit__radiance() {
    const HitGroupData* data = (const HitGroupData*)optixGetSbtDataPointer();
    const float2 barycentrics = optixGetTriangleBarycentrics();
    float2 texCoords = getTexCoords(data, barycentrics);
    float3 normal = getNormal(data, barycentrics);
    
    // Material properties
    float3 albedo = data->material.albedo;
    float metallic = data->material.metallic;
    float roughness = data->material.roughness;
    
    // Sample textures if available
    if (data->albedo_texture) {
        float4 texColor = sampleTexture(data->albedo_texture, texCoords);
        albedo = make_float3(texColor.x, texColor.y, texColor.z);
    }
    
    if (data->metallic_roughness_texture) {
        float4 mrTexColor = sampleTexture(data->metallic_roughness_texture, texCoords);
        // In most PBR textures:
        // - Metallic is in the blue channel
        // - Roughness is in the green channel
        metallic = mrTexColor.z;
        roughness = mrTexColor.y;
    }

    if (data->normal_texture && data->normals) {  // Just check for normals directly
        float4 normalTexColor = sampleTexture(data->normal_texture, texCoords);
        float3 tangentSpaceNormal = make_float3(
            normalTexColor.x * 2.0f - 1.0f,
            normalTexColor.y * 2.0f - 1.0f,
            normalTexColor.z * 2.0f - 1.0f
        );
        
        // For visualization, we'll just use the RGB directly for now
        // In a real shader, we would transform this using a TBN matrix
        normal = normalize(tangentSpaceNormal);
    }
    
    // Get hit position and view direction for lighting calculation
    float3 ray_dir = optixGetWorldRayDirection();
    float ray_t = optixGetRayTmax();
    float3 position = optixGetWorldRayOrigin() + ray_t * ray_dir;
    float3 view_dir = -ray_dir;  // Direction from hit point to camera
    
    // Choose visualization mode - Uncomment the one you want to use
    
    // Option 1: Apply lighting to the material (recommended)
    float3 color = calculateLighting(position, normal, view_dir, albedo, metallic, roughness);
    
    // Option 2: Display only albedo (base color)
    // float3 color = albedo;
    
    // Option 3: Display normals mapped from [-1,1] to [0,1]
    // float3 color = make_float3(normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f);
    
    // Option 4: Display PBR properties in RGB channels
    // float3 color = make_float3(roughness, roughness, metallic);
    
    // Option 5: Mix albedo with metallic and roughness for debugging
    // color = albedo;
    // color.z = color.z * 0.5f + metallic * 0.5f;  // Add metallic to blue channel
    // color.y = color.y * 0.5f + roughness * 0.5f; // Add roughness to green channel
    
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}