//#include <optix.h>
//#include <optix_device.h>
//#include "../optixDynamicGeometry.h"
//#include <cuda_runtime.h> // Include for basic CUDA functions like dot, normalize, etc.
//#include <vector_functions.h> // Include for vector operations like dot, normalize, reflect, etc.
#include "../common.h"
extern "C" __constant__ Params params;

__device__ __inline__ float3 refract(const float3& I, const float3& N, float eta) {
    float cosI = dot(I, N) * -1.0f;
    float sinT2 = eta * eta * (1.0f - cosI * cosI);
    if (sinT2 > 1.0f) return make_float3(0.0f, 0.0f, 0.0f); // Total internal reflection
    float cosT = sqrtf(1.0f - sinT2);
    return eta * I + (eta * cosI - cosT) * N;
}

__device__ __inline__ float fresnelSchlick(float cosTheta, float eta) {
    float R0 = (1.0f - eta) / (1.0f + eta);
    R0 = R0 * R0;
    return R0 + (1.0f - R0) * powf(1.0f - cosTheta, 5.0f);
}
constexpr unsigned int SBT_STRIDE_COLLAPSE = 0;
static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                prd,
        uint32_t               depth
)
{
    unsigned int p0, p1, p2, p3;
    p0 = __float_as_uint( prd->x );
    p1 = __float_as_uint( prd->y );
    p2 = __float_as_uint( prd->z );
    p3 = depth;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            SBT_STRIDE_COLLAPSE, // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2, p3);
    prd->x = __uint_as_float( p0 );
    prd->y = __uint_as_float( p1 );
    prd->z = __uint_as_float( p2 );
    depth = p3;
}


static __forceinline__ __device__ void setPayload( float3 p, unsigned int depth )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
    optixSetPayload_3(depth);
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            __uint_as_float( optixGetPayload_0() ),
            __uint_as_float( optixGetPayload_1() ),
            __uint_as_float( optixGetPayload_2() )
    );
}
static __forceinline__ __device__ unsigned int getDepth() {
    return optixGetPayload_3();
}

extern "C" __global__ void __raygen__rg() {
    const RayGenData* rg_data = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const float2      d = 2.0f * make_float2(
                                    static_cast< float >( idx.x ) / static_cast< float >( dim.x ),
                                    static_cast< float >( idx.y ) / static_cast< float >( dim.y )
                                            ) - 1.0f;
    unsigned int depth = 0;
    const float3 ray_dir = normalize( d.x * U + d.y * V + W );
    float3 prd = make_float3(0.5f, 0.5f, 0.5f);

    trace(params.handle, eye, ray_dir, 0.00f, 1e16f, &prd, depth);

    params.frame_buffer[idx.y * params.width + idx.x] = make_color( prd );
}

extern "C" __global__ void __closesthit__ch() {
    const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    // fetch current triangle vertices
    float3 data[3];

    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
                               optixGetRayTime(), data );
    unsigned int depth = getDepth();
    float3 diffuse_color = hit_data->color;
    if (hit_data->texture != 0) {
//printf("texture: %d %p\n", hit_data->texture, hit_data->texture_ptr);
        // Retrieve barycentric coordinates set by the intersection shader
        float2 barycentrics = optixGetTriangleBarycentrics();
        float alpha = 1.0f - barycentrics.x - barycentrics.y;
        float beta = barycentrics.x;
        float gamma = barycentrics.y;

        // Get the index of the intersected triangle
        unsigned int primitive_index = optixGetPrimitiveIndex();

        // Retrieve vertex indices from the index buffer
        uint3 *index_buffer = reinterpret_cast<uint3 *>(hit_data->index_buffer);

        uint3 vertex_indices = make_uint3(0, 0, 0);
        vertex_indices= index_buffer[primitive_index];
        // Fetch UVs for each vertex of the triangle using the indices
        float2 *uv_buffer = reinterpret_cast<float2 *>(hit_data->uv_buffer);
        float2 uv0 = make_float2(0, 0);
        float2 uv1 = make_float2(0, 0);
        float2 uv2 = make_float2(0, 0);

        uv0 = uv_buffer[vertex_indices.x];
        uv1 = uv_buffer[vertex_indices.y];
        uv2 = uv_buffer[vertex_indices.z];

        // Interpolate UV coordinates using barycentric weights
        float2 uv = alpha * uv0 + beta * uv1 + gamma * uv2;
        // Now, use the UV coordinates for further processing (e.g., texture sampling)
        float4 tex_color = tex2D<float4>(hit_data->texture, clamp(uv.x, 0.f, 1.f), clamp(uv.y, 0.f, 1.f));
        //printf("uv.x: %f uv.y: %f tex.x: %f tex.y: %f tex.z: %f tex.w: %f\n", uv.x, uv.y, tex_color.x, tex_color.y, tex_color.z, tex_color.w);
        diffuse_color = make_float3(tex_color.x, tex_color.y, tex_color.z);

        setPayload(diffuse_color, depth);
        return ;
    }
    // compute triangle normal
    data[1] -= data[0];
    data[2] -= data[0];
    float3 normal = make_float3(
            data[1].y*data[2].z - data[1].z*data[2].y,
            data[1].z*data[2].x - data[1].x*data[2].z,
            data[1].x*data[2].y - data[1].y*data[2].x );
    float3 ray_dir = optixGetWorldRayDirection();


    // Add point light properties
    float3 light_position = make_float3(10.0f, 10.0f, 10.0f); // Example light position
    float3 light_color = make_float3(1.0f, 1.0f, 1.0f); // White light
    float light_intensity = 5000.0f; // Intensity of the point light

    float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    float3 light_dir = normalize(light_position - hit_point);
    float3 view_dir = normalize(-ray_dir);

    // Diffuse component

    float diffuse_factor = fmaxf(dot(normal, light_dir), 0.0f);
    float3 diffuse = diffuse_color * light_color * diffuse_factor * light_intensity;

    // Specular component (Phong reflection model)
    float3 reflection = reflect(-light_dir, normal);
    float spec_factor = powf(fmaxf(dot(view_dir, reflection), 0.0f), 64); // 64 is the shininess factor
    float3 specular = light_color * spec_factor * light_intensity;

    // Check if the material is glass based on a flag
    if (hit_data->is_glass && depth < 8) {
        float eta = hit_data->IOR;
        float cosTheta = dot(-ray_dir, normal);
        float fresnel = fresnelSchlick(cosTheta, eta);

        float3 reflection_dir = reflect(ray_dir, normal);
        float3 refraction_dir = refract(ray_dir, normal, eta);

        float3 reflection_color = make_float3(0.0f);
        float3 refraction_color = make_float3(0.0f);
        float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
        float3 new_origin = hit_point + 1e-3f * normal; // O
        // Trace reflection ray
        if (fresnel > 0.0f) {
            trace(params.handle, new_origin, reflection_dir, 0.01f, 1e16f, &reflection_color, depth+1);
        }

        // Trace refraction ray
        if (fresnel < 1.0f && length(refraction_dir) > 0.0f) {
            trace(params.handle, new_origin, refraction_dir, 0.01f, 1e16f, &refraction_color, depth+1);
        }

        // Mix reflection and refraction using Fresnel to enhance transparency
        float3 final_color = (1.0f - fresnel) * refraction_color + fresnel * reflection_color;
        final_color = clamp(final_color, 0.0f, 1.0f); // Clamp to ensure valid color values.

        // Ensure some transparency by adding a blending factor
        float transparency = 0.6f; // Adjust this value to make the glass more or less transparent
        float3 transmitted_color = transparency * refraction_color;
        final_color = final_color * (1.0f - transparency) + transmitted_color;
        // Add diffuse and specular components for light interaction
        final_color += diffuse + specular;
        final_color = clamp(final_color, 0.0f, 1.0f); // Ensure the color is in a valid range

        setPayload(final_color, depth);
    } else {
        // For non-glass materials, use a simple diffuse shading or another material response

        float3 final_color = diffuse + specular;
        final_color = clamp(final_color, 0.0f, 1.0f); // Ensure the color is in a valid range

        setPayload(final_color, depth);
    }
}

extern "C" __global__ void __miss__ms() {
    const MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    float3    payload = getPayload();
    setPayload(make_float3(miss_data->bg_color.x, 0.5f, miss_data->bg_color.z), getDepth());
}
