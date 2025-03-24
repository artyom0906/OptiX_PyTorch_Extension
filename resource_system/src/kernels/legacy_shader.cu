#include <optix.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "material_types.h"

// Include the dynamic geometry header from the original OptiX extension
#include "../../../src/optixDynamicGeometry.h"

using namespace optix_renderer;

// Helper functions for vector operations
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

__device__ float3 normalize(const float3& v) {
    float invLen = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

// Adapted legacy shader structures for compatibility
extern "C" {
__constant__ Params params;
}

// Random number generator for the shader
__device__ float rnd(unsigned int& seed) {
    seed = (1664525u * seed + 1013904223u);
    return float(seed & 0x00FFFFFF) / float(0x01000000);
}

// Ray generation shader using camera parameters and legacy structures
extern "C" __global__ void __raygen__legacy() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    // Get camera parameters
    float3 camera_origin = params.eye;
    float3 camera_u = params.U;
    float3 camera_v = params.V;
    float3 camera_w = params.W;
    
    // Compute ray parameters
    const float2 d = make_float2(
        float(idx.x) / float(dim.x),
        float(idx.y) / float(dim.y)
    );
    
    // Initialize random seed
    unsigned int seed = params.subframe_index * dim.x * dim.y + idx.y * dim.x + idx.x;
    
    // Initialize radiance
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    
    // Anti-aliasing samples
    for (unsigned int sample = 0; sample < params.samples_per_launch; ++sample) {
        // Jitter pixel location for anti-aliasing
        float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
        float2 subpixel = make_float2(
            d.x + subpixel_jitter.x / dim.x,
            d.y + subpixel_jitter.y / dim.y
        );
        
        // Calculate ray direction through the pixel
        float3 ray_direction = normalize(
            camera_u * (subpixel.x - 0.5f) +
            camera_v * (subpixel.y - 0.5f) +
            camera_w
        );
        
        // Initialize ray payload
        RadiancePRD prd;
        prd.attenuation = make_float3(1.0f, 1.0f, 1.0f);
        prd.seed = seed;
        prd.depth = 0;
        prd.emitted = make_float3(0.0f, 0.0f, 0.0f);
        prd.radiance = make_float3(0.0f, 0.0f, 0.0f);
        prd.origin = camera_origin;
        prd.direction = ray_direction;
        prd.done = false;
        prd.inside = false;
        
        // Trace the ray
        int iters = 0;
        int max_iters = 10; // Prevent infinite loops
        
        while (!prd.done && iters++ < max_iters) {
            // Set up ray payload semantics
            optixTrace(
                params.handle,
                prd.origin,
                prd.direction,
                0.001f,              // tmin
                1e16f,               // tmax
                0.0f,                // ray time
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT, 
                0,                   // SBT offset
                1,                   // SBT stride
                0,                   // missSBTIndex
                // Legacy structure: All ray payload in one structure
                OPTIX_PAYLOAD_TYPE_RADIANCE, &prd
            );
        }
        
        // Accumulate result
        result = result + prd.radiance;
    }
    
    // Average all samples
    result = result / static_cast<float>(params.samples_per_launch);
    
    // Write to output buffer - accumulating over frames for progressive rendering
    uint3 launch_index = optixGetLaunchIndex();
    unsigned int image_index = launch_index.y * params.width + launch_index.x;
    
    if (params.subframe_index > 0) {
        // Progressive refinement: blend with previous frames
        float a = 1.0f / static_cast<float>(params.subframe_index + 1);
        float3 prev_color = make_float3(
            params.accum_buffer[image_index].x,
            params.accum_buffer[image_index].y,
            params.accum_buffer[image_index].z
        );
        result = prev_color * (1.0f - a) + result * a;
    }
    
    // Write accumulated result
    params.accum_buffer[image_index] = make_float4(result.x, result.y, result.z, 1.0f);
    
    // Convert to output format (0-255 RGBA)
    float r = min(result.x, 1.0f);
    float g = min(result.y, 1.0f);
    float b = min(result.z, 1.0f);
    
    params.frame_buffer[image_index] = make_uchar4(
        static_cast<unsigned char>(r * 255.99f),
        static_cast<unsigned char>(g * 255.99f),
        static_cast<unsigned char>(b * 255.99f),
        255
    );
}

// Legacy miss shader that returns a skybox color
extern "C" __global__ void __miss__legacy() {
    MissData* miss_data = (MissData*)optixGetSbtDataPointer();
    
    // Legacy payload handling
    RadiancePRD* prd = getPRD<RadiancePRD>();
    prd->done = true;
    
    // Simple skybox color based on ray direction
    float3 direction = normalize(optixGetWorldRayDirection());
    float t = 0.5f * (direction.y + 1.0f);
    
    // Sky gradient
    float3 sky_color = make_float3(0.5f, 0.7f, 1.0f) * (1.0f - t) + make_float3(0.1f, 0.2f, 0.3f) * t;
    
    // If a background color is specified in the miss data, use it
    if (miss_data) {
        sky_color = make_float3(miss_data->bg_color.x, miss_data->bg_color.y, miss_data->bg_color.z);
    }
    
    prd->radiance = sky_color;
    prd->emitted = sky_color;
}

// Get reflected ray direction
__device__ float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

// Calculate refraction direction
__device__ float3 refract(const float3& v, const float3& n, float ior) {
    float cos_theta = min(dot(-v, n), 1.0f);
    float3 r_perp = ior * (v + cos_theta * n);
    float3 r_parallel = -sqrt(1.0f - dot(r_perp, r_perp)) * n;
    return r_perp + r_parallel;
}

// Fresnel calculation for glass
__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

// Legacy closest hit shader with glass material support
extern "C" __global__ void __closesthit__legacy() {
    // Get hit group data from SBT
    HitGroupData* hit_data = (HitGroupData*)optixGetSbtDataPointer();
    RadiancePRD* prd = getPRD<RadiancePRD>();
    
    // Get intersection information
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const int primitive_idx = optixGetPrimitiveIndex();
    
    // Get the triangle vertices
    int3 index;
    if (hit_data->index_buffer) {
        index = *((int3*)hit_data->index_buffer + primitive_idx);
    } else {
        // Direct indexing if no index buffer
        index.x = 3 * primitive_idx;
        index.y = 3 * primitive_idx + 1;
        index.z = 3 * primitive_idx + 2;
    }
    
    // Get hit position and normal
    float3 p0 = ((float3*)hit_data->vertices)[index.x];
    float3 p1 = ((float3*)hit_data->vertices)[index.y];
    float3 p2 = ((float3*)hit_data->vertices)[index.z];
    
    float3 hit_point = (1.0f - barycentrics.x - barycentrics.y) * p0 + 
                       barycentrics.x * p1 + 
                       barycentrics.y * p2;
    
    float3 normal;
    if (hit_data->normals) {
        // Interpolate normals if available
        float3 n0 = ((float3*)hit_data->normals)[index.x];
        float3 n1 = ((float3*)hit_data->normals)[index.y];
        float3 n2 = ((float3*)hit_data->normals)[index.z];
        normal = normalize((1.0f - barycentrics.x - barycentrics.y) * n0 + 
                          barycentrics.x * n1 + 
                          barycentrics.y * n2);
    } else {
        // Otherwise compute flat normal
        normal = normalize(cross(p1 - p0, p2 - p0));
    }
    
    // Get UV coordinates if available
    float2 uv = make_float2(0.0f, 0.0f);
    if (hit_data->uv_buffer) {
        float2 uv0 = ((float2*)hit_data->uv_buffer)[index.x];
        float2 uv1 = ((float2*)hit_data->uv_buffer)[index.y];
        float2 uv2 = ((float2*)hit_data->uv_buffer)[index.z];
        uv = (1.0f - barycentrics.x - barycentrics.y) * uv0 + 
             barycentrics.x * uv1 + 
             barycentrics.y * uv2;
    }
    
    // Ensure normal is pointing outward
    float3 ray_dir = optixGetWorldRayDirection();
    if (dot(normal, ray_dir) > 0.0f) {
        normal = -normal;
    }
    
    // Handle glass material
    if (hit_data->is_glass) {
        float refraction_ratio = prd->inside ? hit_data->IOR : 1.0f / hit_data->IOR;
        
        // Calculate Fresnel reflection probability
        float cos_theta = min(dot(-ray_dir, normal), 1.0f);
        float reflect_prob = schlick(cos_theta, refraction_ratio);
        
        // Either reflect or refract based on probability
        float3 scattered_dir;
        if (rnd(prd->seed) < reflect_prob) {
            // Reflect
            scattered_dir = reflect(ray_dir, normal);
        } else {
            // Refract
            scattered_dir = refract(ray_dir, normal, refraction_ratio);
            prd->inside = !prd->inside;
        }
        
        // Set up next ray
        prd->origin = hit_point + 0.001f * (prd->inside ? -normal : normal);
        prd->direction = scattered_dir;
        
        // Glass transmits all light (no attenuation) but can be tinted
        prd->attenuation = prd->attenuation * hit_data->color;
        
        // Add emission if any
        prd->emitted = hit_data->emission;
        
    } else {
        // Standard Lambertian material
        float3 color = hit_data->color;
        
        // Use texture if available
        if (hit_data->texture && hit_data->uv_buffer) {
            float4 tex_color = tex2D<float4>(hit_data->texture, uv.x, uv.y);
            color = make_float3(tex_color.x, tex_color.y, tex_color.z);
        }
        
        // Check if we've reached max depth
        if (prd->depth >= params.max_depth) {
            prd->radiance = make_float3(0.0f);
            prd->done = true;
            return;
        }
        
        // Lambertian scatter
        float3 scatter_direction = normal + normalize(make_float3(
            2.0f * rnd(prd->seed) - 1.0f,
            2.0f * rnd(prd->seed) - 1.0f,
            2.0f * rnd(prd->seed) - 1.0f
        ));
        
        // Avoid degenerate scatter direction
        if (dot(scatter_direction, scatter_direction) < 0.001f)
            scatter_direction = normal;
            
        scatter_direction = normalize(scatter_direction);
        
        // Set up next ray
        prd->origin = hit_point + 0.001f * normal;
        prd->direction = scatter_direction;
        prd->attenuation = prd->attenuation * color;
        prd->depth++;
        
        // Add emission if any
        if (hit_data->emission_texture) {
            float4 emit_tex = tex2D<float4>(hit_data->emission_texture, uv.x, uv.y);
            prd->emitted = make_float3(emit_tex.x, emit_tex.y, emit_tex.z);
        } else {
            prd->emitted = hit_data->emission;
        }
    }
}

// Legacy any hit shader - needed for shadows with transparent objects
extern "C" __global__ void __anyhit__legacy() {
    HitGroupData* hit_data = (HitGroupData*)optixGetSbtDataPointer();
    
    // For transparent materials (glass), potentially continue the ray
    if (hit_data->is_glass) {
        // Simple attenuation - could be more sophisticated
        RadiancePRD* prd = getPRD<RadiancePRD>();
        prd->attenuation = prd->attenuation * hit_data->color;
        
        // Continue the ray (ignore this hit)
        optixIgnoreIntersection();
    }
}