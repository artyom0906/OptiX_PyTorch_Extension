#define OPTIX_USE_LEGACY_NAMES 0
#include "../sutil/helper_math.h"
#include <optix.h>
#include <optix_device.h>

struct Params {
    float3* image;
    int width;
    int height;
    float3 eye;
    float3 U;
    float3 V;
    float3 W;
    OptixTraversableHandle handle;
};

extern "C" {
__constant__ Params params;
}

// Ray payload structure
struct Payload {
    float3 color;
};

// Utility functions to pack and unpack payload pointers
static __forceinline__ __device__ void packPointer( void* ptr, unsigned int& u0, unsigned int& u1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    u0 = static_cast<unsigned int>( uptr );
    u1 = static_cast<unsigned int>( uptr >> 32 );
}

static __forceinline__ __device__ Payload* getPayload()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    const unsigned long long uptr = ( static_cast<unsigned long long>( u1 ) << 32 ) | u0;
    return reinterpret_cast<Payload*>( uptr );
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Compute normalized screen coordinates [0,1]
    float2 d = make_float2(
                       (static_cast<float>(idx.x) + 0.5f) / static_cast<float>(dim.x),
                       (static_cast<float>(idx.y) + 0.5f) / static_cast<float>(dim.y)
                               ) * 2.0f - 1.0f;

    // Compute ray direction
    float3 ray_origin = params.eye;
    float3 ray_direction = normalize(d.x * params.U + d.y * params.V - params.W);

    // Initialize payload
    Payload payload;
    payload.color = make_float3(0.0f);

    // Trace the ray
    unsigned int u0, u1;
    packPointer(&payload, u0, u1);
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // tmin
            1e16f,               // tmax
            0.0f,                // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            1,                   // SBT stride
            0,                   // missSBTIndex
            u0, u1
    );

    // Write the color to the output image
    int pixel_idx = idx.y * params.width + idx.x;
    params.image[pixel_idx] = payload.color;
}

// Closest hit program
extern "C" __global__ void __closesthit__ch() {
    // Get the payload
    Payload* payload = getPayload();

    // Simple shading: color the sphere red
    payload->color = make_float3(1.0f, 0.0f, 0.0f);
}

// Miss program
extern "C" __global__ void __miss__ms() {
    // Get the payload
    Payload* payload = getPayload();

    // Background color (sky blue)
    payload->color = make_float3(0.7f, 0.8f, 1.0f);
}


