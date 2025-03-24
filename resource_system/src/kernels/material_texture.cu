#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>  // For CUtexObject
#include <vector_types.h>
#include <vector_functions.h>
#include <math_constants.h>  // For CUDA math constants
#include "material_types.h"  // Include the enhanced material types

// Define M_PI if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

__device__ inline float3 operator*(float a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 normalize(const float3& v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

__device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Removed to avoid duplicate mix functions

// The hit group data structure must match the one in LaunchParams.h
struct HitGroupData {
    // Material data
    SimpleMaterial material;
    
    // Geometry data
    CUdeviceptr vertices;
    CUdeviceptr indices;
    CUdeviceptr normals;
    CUdeviceptr texcoords;
    CUdeviceptr tangents;
    CUdeviceptr bitangents;
    
    // Texture objects - using CUtexObject for CUDA driver API compatibility
    CUtexObject albedo_texture;
    CUtexObject normal_texture;
    CUtexObject metallic_roughness_texture;
    CUtexObject emission_texture;
    CUtexObject specular_texture;
    CUtexObject specular_tint_texture;
    CUtexObject sheen_texture;
    CUtexObject clearcoat_texture;
    
    // Flags for geometry features
    bool has_normals;
    bool has_texcoords;
    bool has_tangents;
    bool has_bitangents;
    
    // Legacy compatibility fields
    float3 albedo;
    float3 emission;
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

// Compute normal at hit point
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
    
    float3 normal;
    if (data->has_normals && data->normals) {
        // Interpolate vertex normals
        float3* normals = (float3*)(data->normals);
        float3 n0 = normals[indices.x];
        float3 n1 = normals[indices.y];
        float3 n2 = normals[indices.z];
        
        float w0 = 1.0f - barycentrics.x - barycentrics.y;
        float w1 = barycentrics.x;
        float w2 = barycentrics.y;
        
        normal = w0 * n0 + w1 * n1 + w2 * n2;
    } else {
        // Compute geometric normal
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        normal = cross(e1, e2);
    }
    
    return normalize(normal);
}

// Get texture coordinates at hit point
__device__ float2 getTexCoords(const HitGroupData* data, const float2& barycentrics) {
    if (!data->has_texcoords || !data->texcoords) {
        return make_float2(0.0f, 0.0f);
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
    return make_float4(1.0f, 1.0f, 1.0f, 1.0f); // Default white
}

// Get tangent space basis vectors
__device__ void getTangentSpace(const HitGroupData* data, const float2& barycentrics,
                               float3& tangent, float3& bitangent) {
    // Default tangent space
    tangent = make_float3(1.0f, 0.0f, 0.0f);
    bitangent = make_float3(0.0f, 1.0f, 0.0f);
    
    if (!data->has_tangents || !data->tangents || !data->has_bitangents || !data->bitangents) {
        return;
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
    
    // Interpolate tangents
    float3* tangents = (float3*)(data->tangents);
    float3 t0 = tangents[indices.x];
    float3 t1 = tangents[indices.y];
    float3 t2 = tangents[indices.z];
    
    // Interpolate bitangents
    float3* bitangents = (float3*)(data->bitangents);
    float3 bt0 = bitangents[indices.x];
    float3 bt1 = bitangents[indices.y];
    float3 bt2 = bitangents[indices.z];
    
    // Interpolate using barycentric coordinates
    float w0 = 1.0f - barycentrics.x - barycentrics.y;
    float w1 = barycentrics.x;
    float w2 = barycentrics.y;
    
    // Component-wise interpolation
    float3 t;
    t.x = w0 * t0.x + w1 * t1.x + w2 * t2.x;
    t.y = w0 * t0.y + w1 * t1.y + w2 * t2.y;
    t.z = w0 * t0.z + w1 * t1.z + w2 * t2.z;
    tangent = normalize(t);
    
    float3 bt;
    bt.x = w0 * bt0.x + w1 * bt1.x + w2 * bt2.x;
    bt.y = w0 * bt0.y + w1 * bt1.y + w2 * bt2.y;
    bt.z = w0 * bt0.z + w1 * bt1.z + w2 * bt2.z;
    bitangent = normalize(bt);
}

// Apply normal mapping
__device__ float3 applyNormalMap(float3 normal, float3 tangent, float3 bitangent, float3 normalMapSample) {
    // Convert from [0,1] to [-1,1] range
    float3 normalFromMap = make_float3(
        normalMapSample.x * 2.0f - 1.0f,
        normalMapSample.y * 2.0f - 1.0f,
        normalMapSample.z * 2.0f - 1.0f
    );
    
    // Ensure normal is unit length
    normalFromMap = normalize(normalFromMap);
    
    // Create the TBN matrix to transform from tangent space to world space
    float3 N = normal;
    float3 T = normalize(tangent - dot(tangent, N) * N); // Ensure T is perpendicular to N
    float3 B = normalize(bitangent - dot(bitangent, N) * N - dot(bitangent, T) * T);
    
    // Transform the normal from the normal map to world space
    return normalize(
        normalFromMap.x * T +
        normalFromMap.y * B +
        normalFromMap.z * N
    );
}

// Schlick approximation for Fresnel term
__device__ float3 schlick(float3 f0, float cosTheta) {
    float p = powf(1.0f - cosTheta, 5.0f);
    return make_float3(
        f0.x + (1.0f - f0.x) * p,
        f0.y + (1.0f - f0.y) * p,
        f0.z + (1.0f - f0.z) * p
    );
}

// GGX Distribution
__device__ float ggxDistribution(float roughness, float NoH) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NoH2 = NoH * NoH;
    float denom = NoH2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (M_PI * denom * denom);
}

// Smith GGX geometry term
__device__ float smithGGX(float roughness, float NoV, float NoL) {
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    float g1V = NoV / (NoV * (1.0f - k) + k);
    float g1L = NoL / (NoL * (1.0f - k) + k);
    return g1V * g1L;
}

// Linear interpolation for float3 (mixed_f3 to avoid conflict)
__device__ float3 mixed_f3(float3 a, float3 b, float t) {
    return make_float3(
        a.x * (1.0f - t) + b.x * t,
        a.y * (1.0f - t) + b.y * t,
        a.z * (1.0f - t) + b.z * t
    );
}

// Linear interpolation for float (mixed_f to avoid conflict)
__device__ float mixed_f(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

// Negate a float3
__device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

// Basic PBR shading function
__device__ float3 evaluatePBR(const SimpleMaterial& material, float3 baseColor, 
                           float3 normal, float3 viewDir) {
    // Simple lighting from the top-right
    float3 lightDir = normalize(make_float3(0.5f, 1.0f, 0.3f));
    float3 lightColor = make_float3(1.0f, 1.0f, 1.0f);
    float lightIntensity = 3.0f;
    
    // Get material properties
    float metallic = material.metallic;
    float roughness = max(0.01f, material.roughness); // Clamp to avoid division by zero
    
    // Calculate necessary dot products
    float NoL = max(0.001f, dot(normal, lightDir));
    float NoV = max(0.001f, dot(normal, viewDir));
    float3 halfVector = normalize(lightDir + viewDir);
    float NoH = max(0.001f, dot(normal, halfVector));
    float VoH = max(0.001f, dot(viewDir, halfVector));
    
    // Specular reflection color (f0)
    float3 f0 = mixed_f3(make_float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
    
    // Calculate specular BRDF terms
    float3 F = schlick(f0, VoH);
    float D = ggxDistribution(roughness, NoH);
    float G = smithGGX(roughness, NoV, NoL);
    
    // Specular and diffuse components
    float spec_denom = 4.0f * NoV * NoL;
    float3 specular = make_float3(
        D * G * F.x / spec_denom,
        D * G * F.y / spec_denom,
        D * G * F.z / spec_denom
    );
    float3 diffuse = make_float3(
        baseColor.x * (1.0f - metallic) * (1.0f / (float)CUDART_PI_F),
        baseColor.y * (1.0f - metallic) * (1.0f / (float)CUDART_PI_F),
        baseColor.z * (1.0f - metallic) * (1.0f / (float)CUDART_PI_F)
    );
    
    // Calculate sheen if enabled
    float3 sheen = make_float3(0.0f, 0.0f, 0.0f);
    if (getMaterialFlag(material.flags, MATERIAL_HAS_SHEEN) && material.sheen > 0.0f) {
        float sheenDistribution = (2.0f + 1.0f) * powf(1.0f - NoH, 5.0f) / (2.0f * CUDART_PI_F);
        float3 sheenColor = mixed_f3(make_float3(1.0f, 1.0f, 1.0f), baseColor, material.sheenTint);
        
        sheen.x = material.sheen * sheenColor.x * sheenDistribution;
        sheen.y = material.sheen * sheenColor.y * sheenDistribution;
        sheen.z = material.sheen * sheenColor.z * sheenDistribution;
    }
    
    // Calculate clearcoat if enabled
    float3 clearcoat = make_float3(0.0f, 0.0f, 0.0f);
    if (getMaterialFlag(material.flags, MATERIAL_HAS_CLEARCOAT) && material.clearcoat > 0.0f) {
        float clearcoatRoughness = mixed_f(0.1f, 0.001f, material.clearcoatGloss);
        float clearcoatD = ggxDistribution(clearcoatRoughness, NoH);
        float clearcoatF = 0.04f + 0.96f * powf(1.0f - VoH, 5.0f);
        
        float cc = material.clearcoat * clearcoatD * clearcoatF;
        clearcoat = make_float3(cc, cc, cc);
    }
    
    // Combine all lighting components
    float3 result;
    result.x = (diffuse.x + specular.x + sheen.x + clearcoat.x) * lightColor.x * NoL * lightIntensity;
    result.y = (diffuse.y + specular.y + sheen.y + clearcoat.y) * lightColor.y * NoL * lightIntensity;
    result.z = (diffuse.z + specular.z + sheen.z + clearcoat.z) * lightColor.z * NoL * lightIntensity;
    
    // Add emission
    if (getMaterialFlag(material.flags, MATERIAL_IS_EMISSIVE)) {
        result.x += material.emission.x;
        result.y += material.emission.y;
        result.z += material.emission.z;
    }
    
    // Add ambient lighting
    float ambient_strength = 0.03f;
    result.x += baseColor.x * ambient_strength;
    result.y += baseColor.y * ambient_strength;
    result.z += baseColor.z * ambient_strength;
    
    return result;
}

// Ray generation program with PBR materials and textures
extern "C" __global__ void __raygen__pbr() {
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
    
    // Default fallback color (background)
    float3 backgroundColor = make_float3(0.15f, 0.15f, 0.15f);
    
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

// Miss shader that returns a basic environment
extern "C" __global__ void __miss__environment() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    // Get ray direction to create a gradient environment map
    float3 rayDir = normalize(optixGetWorldRayDirection());
    
    // Simple sky gradient based on ray Y direction
    float t = 0.5f * (rayDir.y + 1.0f);
    float3 color = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
    
    // Set the payload to the computed color
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

// Closest hit shader with enhanced material support
extern "C" __global__ void __closesthit__pbr() {
    // Get the SBT record data to access the material for this object
    const HitGroupData* data = (const HitGroupData*)optixGetSbtDataPointer();
    const SimpleMaterial& material = data->material;
    
    // Get hit information
    const float2 barycentrics = optixGetTriangleBarycentrics();
    float3 normal = computeNormal(data, barycentrics);
    float3 worldPosition = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
    // Manually negate the direction vector to avoid operator- issues
    float3 rayDir = optixGetWorldRayDirection();
    float3 viewDirection = make_float3(-rayDir.x, -rayDir.y, -rayDir.z);
    viewDirection = normalize(viewDirection);
    
    // Get texture coordinates
    float2 texCoords = getTexCoords(data, barycentrics);
    
    // Prepare for normal mapping
    float3 tangent = make_float3(1.0f, 0.0f, 0.0f);
    float3 bitangent = make_float3(0.0f, 1.0f, 0.0f);
    getTangentSpace(data, barycentrics, tangent, bitangent);
    
    // Apply normal mapping if available
    if (getMaterialFlag(material.flags, MATERIAL_HAS_NORMAL_TEXTURE) && 
        material.normalTexture >= 0 && data->normal_texture) {
        float4 normalSample = sampleTexture(data->normal_texture, texCoords);
        normal = applyNormalMap(normal, tangent, bitangent, 
                               make_float3(normalSample.x, normalSample.y, normalSample.z));
    }
    
    // Fetch base color (albedo)
    float3 baseColor = material.albedo;
    if (getMaterialFlag(material.flags, MATERIAL_HAS_ALBEDO_TEXTURE) && 
        material.albedoTexture >= 0 && data->albedo_texture) {
        float4 albedoSample = sampleTexture(data->albedo_texture, texCoords);
        baseColor = make_float3(albedoSample.x, albedoSample.y, albedoSample.z);
    }
    
    // Sample metallic/roughness if available
    float metallic = material.metallic;
    float roughness = material.roughness;
    if (getMaterialFlag(material.flags, MATERIAL_HAS_METALLIC_ROUGHNESS_TEXTURE) && 
        material.metallicRoughnessTexture >= 0 && data->metallic_roughness_texture) {
        float4 mrSample = sampleTexture(data->metallic_roughness_texture, texCoords);
        metallic = mrSample.x;   // Metallic stored in R channel
        roughness = mrSample.y;  // Roughness stored in G channel
    }
    
    // Sample emission if needed
    float3 emission = material.emission;
    if (getMaterialFlag(material.flags, MATERIAL_HAS_EMISSION_TEXTURE) && 
        material.emissionTexture >= 0 && data->emission_texture) {
        float4 emissionSample = sampleTexture(data->emission_texture, texCoords);
        emission = make_float3(emissionSample.x, emissionSample.y, emissionSample.z);
        
        // Can't modify const material, just use the emission value directly below
    }
    
    // Handle different material types
    float3 color;
    
    switch (material.materialType) {
        case OPTIX_LAMBERTIAN: {
            // Simple diffuse shading
            float ndotl = max(0.0f, dot(normal, normalize(make_float3(0.5f, 1.0f, 0.3f))));
            color = baseColor * (0.2f + 0.8f * ndotl); // 0.2 ambient + 0.8 direct
            break;
        }
        
        case OPTIX_PBR: {
            // Physically based shading
            color = evaluatePBR(material, baseColor, normal, viewDirection);
            break;
        }
        
        case OPTIX_EMISSIVE: {
            // Emissive materials just emit light
            color = emission;
            break;
        }
        
        case OPTIX_GLASS:
        case OPTIX_MIRROR:
        case OPTIX_MDL:
            // These advanced material types will be implemented in later phases
            // For now just use a distinctive color
            color = make_float3(1.0f, 0.0f, 1.0f); // Magenta for unimplemented materials
            break;
            
        default:
            // Fallback for unrecognized material types
            color = baseColor;
            break;
    }
    
    // Set the payload values to our computed color
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}