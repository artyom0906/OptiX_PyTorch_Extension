#include <optix.h>
#include <cuda_runtime.h>

// Vector operations - needed for CUDA
__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 normalize(const float3& v) {
    float invLen = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

// Simple launch parameters
struct Params {
    // Camera information
    float3 camera_position;
    float3 camera_u;
    float3 camera_v;
    float3 camera_w;

    // Dimensions
    int width;
    int height;
    int samples_per_pixel;
    int max_depth;

    // Output buffer
    CUdeviceptr output_buffer;

    // Scene handle
    OptixTraversableHandle traversable;
};

// Ray generation shader - starting point for each ray
extern "C" __global__ void __raygen__renderFrame() {
    // Get pixel coordinates
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    // Get launch parameters
    const Params* params = (Params*)optixGetSbtDataPointer();
    
    // Calculate normalized device coordinates (0-1)
    const float2 pixel = make_float2(idx.x + 0.5f, idx.y + 0.5f);
    const float2 d = make_float2(pixel.x / params->width, pixel.y / params->height);
    
    // Transform from [0,1] to [-1,1]
    const float2 screen = make_float2(
         2.0f * static_cast< float >( idx.x ) / static_cast< float >( dim.x ) - 1.0f,
         2.0f * static_cast< float >( idx.y ) / static_cast< float >( dim.y ) - 1.0f
    );
    
    // Calculate ray direction from camera parameters
    const float3 ray_origin = params->camera_position;
    const float3 ray_direction = normalize(
        params->camera_u * screen.x +
        params->camera_v * screen.y +
        params->camera_w
    );
    
    // Initialize ray payload variables
    unsigned int p0 = 0;  // Red
    unsigned int p1 = 0;  // Green
    unsigned int p2 = 0;  // Blue
    
    // Trace the ray
    optixTrace(
        params->traversable,
        ray_origin,
        ray_direction,
        0.0f,                  // Min distance
        1e16f,                 // Max distance
        0.0f,                  // Ray time (for motion blur)
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                     // SBT offset
        1,                     // SBT stride
        0,                     // Miss SBT index
        p0, p1, p2             // Payload (RGB values)
    );
    
    // Convert payload to float RGB values
    float r = __uint_as_float(p0);
    float g = __uint_as_float(p1);
    float b = __uint_as_float(p2);
    
    // Write to output buffer
    float4* output = (float4*)params->output_buffer;
    output[idx.y * params->width + idx.x] = make_float4(r, g, b, 1.0f);
}

// Miss shader - called when ray doesn't hit any geometry
extern "C" __global__ void __miss__environment() {
    // Simply return dark gray color for the background
    float r = 0.2f;
    float g = 0.2f; 
    float b = 0.2f;
    
    optixSetPayload_0(__float_as_uint(r));
    optixSetPayload_1(__float_as_uint(g));
    optixSetPayload_2(__float_as_uint(b));
}

// Closest hit shader - called when ray hits geometry
extern "C" __global__ void __closesthit__radiance() {
    // Get primitive index (which triangle we hit)
    const int primitiveIndex = optixGetPrimitiveIndex();
    
    // Simple color scheme: different color for each face
    float3 color;
    switch (primitiveIndex % 12) {
        case 0:
        case 1:
            // Front face - red
            color = make_float3(1.0f, 0.0f, 0.0f);
            break;
        case 2:
        case 3:
            // Right face - green
            color = make_float3(0.0f, 1.0f, 0.0f);
            break;
        case 4:
        case 5:
            // Back face - blue
            color = make_float3(0.0f, 0.0f, 1.0f);
            break;
        case 6:
        case 7:
            // Left face - yellow
            color = make_float3(1.0f, 1.0f, 0.0f);
            break;
        case 8:
        case 9:
            // Top face - cyan
            color = make_float3(0.0f, 1.0f, 1.0f);
            break;
        case 10:
        case 11:
            // Bottom face - magenta
            color = make_float3(1.0f, 0.0f, 1.0f);
            break;
        default:
            // White (shouldn't happen)
            color = make_float3(1.0f, 1.0f, 1.0f);
            break;
    }
    
    // Return the color
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}
