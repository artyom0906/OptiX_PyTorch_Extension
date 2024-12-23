//#include <optix.h>
//#include <optix_device.h>
//#include "../optixDynamicGeometry.h"
//#include <cuda_runtime.h> // Include for basic CUDA functions like dot, normalize, etc.
//#include <vector_functions.h> // Include for vector operations like dot, normalize, reflect, etc.
#include "../common.h"

extern "C" __constant__ Params params;
#define MAX_DEPTH 5
#define MIN_DEPTH 3
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
__device__ float3 fresnelSchlickf3(float cosTheta, float3 F0) {
    return F0 + (make_float3(1.0f) - F0) * powf(1.0f - cosTheta, 5.0f);
}

__device__ float3 cosine_sample_hemisphere(const float3& normal, curandState_t * state) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);

    float r = sqrtf(u1);
    float theta = 2.0f * M_PI * u2;

    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(1.0f - u1);

    // Build TBN matrix
    float3 T, B, N;
    N = normalize(normal);
    if (fabsf(N.x) > fabsf(N.y)) {
        T = normalize(cross(N, make_float3(0.0f, 1.0f, 0.0f)));
    } else {
        T = normalize(cross(N, make_float3(1.0f, 0.0f, 0.0f)));
    }
    B = cross(N, T);

    // Transform sampled direction to world space
    float3 sampled_dir = normalize(x * T + y * B + z * N);
    return sampled_dir;
}



// Sample the texture
__device__ float4 sampleTexture(cudaTextureObject_t texture, const float2 &uv) {
    float4 texColor = tex2D<float4>(texture, uv.x, uv.y);
    return texColor;
}
__device__ float3 safe_normalize(const float3& v) {
    float len = length(v);
    if (len > 1e-6f) {
        return v / len;
    } else {
        // Return a default normal if input is too small
        return make_float3(0.0f, 1.0f, 0.0f);
    }
}
__device__ float D_GGX(const float3& N, const float3& H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = M_PI * denom * denom;

    return a2 / denom;
}
__device__ float G_SchlickGGX(float NdotX, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0f;

    return NdotX / (NdotX * (1.0f - k) + k);
}

__device__ float G_Smith(const float3& N, const float3& V, const float3& L, float roughness) {
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float NdotL = fmaxf(dot(N, L), 0.0f);

    float ggx1 = G_SchlickGGX(NdotV, roughness);
    float ggx2 = G_SchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// Linear interpolation between two floats
__device__ __inline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float3 sampleBRDF(
        const float3& normal,        // Surface normal
        const float3& view_dir,      // Direction towards the camera/viewer
        const float metallic,        // Metallic property of the material
        const float roughness,       // Roughness property for specular
        curandState_t* state,        // RNG state
        float& pdf                   // Output PDF
) {
    // Determine whether to sample diffuse or specular based on material properties
    float rand_val = curand_uniform(state);
    float diffuse_weight = (1.0f - metallic) * (1.0f - roughness);
    float specular_weight = metallic * roughness;
    float total_weight = diffuse_weight + specular_weight;
    float prob = total_weight > 0.0f ? diffuse_weight / total_weight : 0.0f;

    if (rand_val < prob) {
        // **Diffuse Sampling**
        // Using cosine-weighted hemisphere sampling
        float u1 = curand_uniform(state);
        float u2 = curand_uniform(state);
        float r = sqrtf(u1);
        float theta = 2.0f * M_PI * u2;

        float x = r * cosf(theta);
        float y = r * sinf(theta);
        float z = sqrtf(1.0f - u1);

        // Orthonormal basis
        float3 T, B;
        if (fabsf(normal.x) > fabsf(normal.y)) {
            T = normalize(cross(normal, make_float3(0.0f, 1.0f, 0.0f)));
        } else {
            T = normalize(cross(normal, make_float3(1.0f, 0.0f, 0.0f)));
        }
        B = cross(normal, T);

        // Sampled direction in world space
        float3 L = normalize(x * T + y * B + z * normal);

        // PDF for cosine-weighted hemisphere
        pdf = dot(normal, L) / M_PI;

        return L;
    } else {
        // **Specular Sampling**
        // GGX importance sampling
        float a = roughness * roughness;
        float phi = 2.0f * M_PI * curand_uniform(state);
        float cosTheta = sqrtf((1.0f - curand_uniform(state)) / (1.0f + (a * a - 1.0f) * curand_uniform(state)));
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

        // Half-vector
        float3 H = normalize(make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta));

        // Reflection direction
        float3 L = reflect(-view_dir, H);
        L = normalize(L);

        // PDF for GGX
        float NdotH = max(dot(normal, H), 0.0f);
        float D = (a * a) / (M_PI * powf((NdotH * NdotH * (a * a - 1.0f) + 1.0f), 2.0f));
        pdf = D * NdotH / (4.0f * max(dot(view_dir, H), 0.0f) + 1e-5f); // Avoid division by zero

        return L;
    }
}

struct PerRayData
{
    float3 accumulated_color; // Accumulated color from all bounces
    float3 throughput;        // The cumulative product of BRDFs and cosines
    uint32_t depth;
    curandState_t  state;        // `curand` RNG state
};


constexpr unsigned int SBT_STRIDE_COLLAPSE = 0;
static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        PerRayData*             prd
)
{
    unsigned int p0, p1, p2, p3;
    size_t prdPtr = reinterpret_cast<size_t>(prd);

    p0 = static_cast<uint32_t>(prdPtr & 0xFFFFFFFF);
    p1 = static_cast<uint32_t>(prdPtr >> 32);
    p2 = 0;
    p3 = 0;

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
}


static __forceinline__ __device__ PerRayData * getPRD(){
    PerRayData *prd = reinterpret_cast<PerRayData*>(
            static_cast<size_t>(optixGetPayload_0()) |
            (static_cast<size_t>(optixGetPayload_1()) << 32)
    );
    return prd;
}

extern "C" __global__ void __raygen__rg()
{
    const int    w   = params.width;
    const int    h   = params.height;
    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;

    unsigned int seed = tea<4>( idx.y*w + idx.x, subframe_index );

    float3 result = make_float3( 0.0f );
    int i = params.samples_per_launch;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2( rnd( seed ), rnd( seed ) );

        const float2 d = 2.0f * make_float2(
                                        ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
                                        ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
                                                ) - 1.0f;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);
        float3 ray_origin    = eye;

        RadiancePRD prd;
        prd.attenuation  = make_float3(1.f);
        prd.seed         = seed;
        prd.depth        = 0;

        for( ;; )
        {
            traceRadiance(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    0.01f,  // tmin       // TODO: smarter offset
                    1e16f,  // tmax
                    prd );

            result += prd.emitted;
            result += prd.radiance * prd.attenuation;

            const float p = dot( prd.attenuation, make_float3( 0.30f, 0.59f, 0.11f ) );
            const bool done = prd.done  || rnd( prd.seed ) > p;
            if( done )
                break;
            prd.attenuation /= p;

            ray_origin    = prd.origin;
            ray_direction = prd.direction;

            ++prd.depth;
        }
    }
    while( --i );

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index  = launch_index.y * params.width + launch_index.x;
    float3         accum_color  = result / static_cast<float>( params.samples_per_launch );

    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }
    params.accum_buffer[ image_index ] = make_float4( accum_color, 1.0f);
    params.frame_buffer[ image_index ] = make_color ( accum_color );
}

extern "C" __global__ void __closesthit__ch() {
    const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    PerRayData *prd = getPRD();

    if (prd->depth >= MAX_DEPTH) {
        //printf("%d\n\r", prd->depth);
        // Reached maximum recursion depth
        return;
    }

    unsigned int primitive_index = optixGetPrimitiveIndex();
    uint3* index_buffer = reinterpret_cast<uint3 *>(hit_data->index_buffer);
    uint3  vertex_indices= index_buffer[primitive_index];

    float2 barycentrics = optixGetTriangleBarycentrics();
    float alpha = 1.0f - barycentrics.x - barycentrics.y;
    float beta = barycentrics.x;
    float gamma = barycentrics.y;

    float2 *uv_buffer = reinterpret_cast<float2 *>(hit_data->uv_buffer);
    float2 uv0 = uv_buffer[vertex_indices.x];
    float2 uv1 = uv_buffer[vertex_indices.y];
    float2 uv2 = uv_buffer[vertex_indices.z];
    float2 uv = alpha * uv0 + beta * uv1 + gamma * uv2;

    float3* normals = reinterpret_cast<float3*>(hit_data->normals);
    float3 n0 = normals[vertex_indices.x];
    float3 n1 = normals[vertex_indices.y];
    float3 n2 = normals[vertex_indices.z];
    float3 interpolated_normal = safe_normalize(alpha * n0 + beta * n1 + gamma * n2);

    float3* tangents = reinterpret_cast<float3*>(hit_data->tangents);
    float3* bitangents = reinterpret_cast<float3*>(hit_data->bitangents);

    float3 t0 = tangents[vertex_indices.x];
    float3 t1 = tangents[vertex_indices.y];
    float3 t2 = tangents[vertex_indices.z];
    float3 bt0 = bitangents[vertex_indices.x];
    float3 bt1 = bitangents[vertex_indices.y];
    float3 bt2 = bitangents[vertex_indices.z];

    float3 interpolated_tangent   = safe_normalize(alpha * t0 + beta * t1 + gamma * t2);
    float3 interpolated_bitangent = safe_normalize(alpha * bt0 + beta * bt1 + gamma * bt2);

    if(hit_data->normal_map != 0){
        float4 texColor  = sampleTexture(hit_data->normal_map, uv);

        // Convert from [0, 1] to [-1, 1]
        float3 tangent_space_normal  = make_float3(
                texColor.x * 2.0f - 1.0f,
                texColor.y * 2.0f - 1.0f,
                texColor.z * 2.0f - 1.0f
        );

        //const float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(interpolated_normal));

        // Build TBN matrix
        float3 T = normalize(interpolated_normal.x > 0.9f ? make_float3(0.0f, 1.0f, 0.0f) : cross(make_float3(1.0f, 0.0f, 0.0f), interpolated_normal));
        float3 B = cross(interpolated_normal, T);

        // Transform to world space
        interpolated_normal = normalize(T * tangent_space_normal.x + B * tangent_space_normal.y + interpolated_normal * tangent_space_normal.z);
    }

    // Sample textures
    float3 base_color = make_float3(1.0f, 1.0f, 1.0f);
    float metallic = 0.0f;
    float roughness = 1.0f;
    float3 emission = make_float3(0.0f);

    if (hit_data->texture != 0) {
        //printf("uv.x: %f uv.y: %f tex.x: %f tex.y: %f tex.z: %f tex.w: %f\n", uv.x, uv.y, tex_color.x, tex_color.y, tex_color.z, tex_color.w);
        float4 texture_color_with_alpha = sampleTexture(hit_data->texture, uv);
        base_color  = make_float3(texture_color_with_alpha.x, texture_color_with_alpha.y, texture_color_with_alpha.z);
        //transparency = texture_color_with_alpha.w;
    }
    if(hit_data->metallic_roughness != 0){
        float4 tex = sampleTexture(hit_data->metallic_roughness, uv);
        metallic = tex.x;
        roughness = tex.y;
    }
    if(hit_data->emission_texture != 0){
        float4 tex = sampleTexture(hit_data->emission_texture, uv);
        emission = make_float3(tex.x, tex.y, tex.z)*10.0f;
    }

    prd->accumulated_color += prd->throughput * emission;


    /*
    // Material properties
    //const float3 base_color = clamp(hit_data->color * texture_color, 0.0, 1.0);

    //blender 1- ((transparency<=0.050)?1.f:0.f)// 1.0 opaque and 0.0 (fully transparent)
    //((transparency<=0.1)?1.f:0.f); // Adjust this between 0.0 (opaque) and 1.0 (fully transparent)
*/
    float transparency_factor = 0.3;
    // Calculate F0 based on metallic property
    float F0_scalar = 0.04f;
    float3 F0 = make_float3(F0_scalar); // Default F0 for dielectrics
    F0 = lerp(F0, base_color, metallic); // For metals, F0 is the base color
    // Calculate cosTheta and determine entering or exiting
    float cosTheta = dot(-optixGetWorldRayDirection(), interpolated_normal);
    bool entering = cosTheta > 0.0f;
    cosTheta = fmaxf(cosTheta, 0.0f);

    // Define indices of refraction
    const float IOR_AIR = 1.0f;
    const float IOR_MATERIAL = hit_data->IOR; // e.g., 1.5 for glass

    // Calculate eta based on the ray's direction
    float eta_ratio;
    if (entering) {
        eta_ratio = IOR_AIR / IOR_MATERIAL;
    } else {
        eta_ratio = IOR_MATERIAL / IOR_AIR;
    }

    if (hit_data->is_glass && prd->depth < MAX_DEPTH) {
        float3 fresnel = fresnelSchlickf3(cosTheta, F0);

        float3 reflection_dir = safe_normalize(reflect(optixGetWorldRayDirection(), interpolated_normal));
        float3 refraction_dir = safe_normalize(refract(optixGetWorldRayDirection(), interpolated_normal, eta_ratio));

        float3 reflection_color = make_float3(0.0f);
        float3 refraction_color = make_float3(0.0f);
        float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
        float3 new_origin = safe_normalize(hit_point + 1e-3f * interpolated_normal);// O
        // Trace reflection ray
        if (fresnel.x > 0.0f || fresnel.y > 0.0f || fresnel.z > 0.0f) {
            PerRayData fresnelPRD;
            fresnelPRD.accumulated_color = make_float3(0.0f);
            fresnelPRD.throughput = prd->throughput * fresnel;
            fresnelPRD.depth = prd->depth + 1;
            fresnelPRD.state = prd->state;
            trace(params.handle, new_origin, reflection_dir, 0.01f, 1e16f, &fresnelPRD);
            reflection_color = fresnelPRD.accumulated_color;
        }

        // Trace refraction ray
        if ((fresnel.x < 0.0f || fresnel.y < 0.0f || fresnel.z < 0.0f) && length(refraction_dir) > 0.0f && (!isnan(refraction_dir.x) && !isnan(refraction_dir.x) && !isnan(refraction_dir.x))) {
            PerRayData fresnelPRD;
            fresnelPRD.accumulated_color = make_float3(0.0f);
            fresnelPRD.throughput = prd->throughput * fresnel;
            fresnelPRD.depth = prd->depth + 1;
            fresnelPRD.state = prd->state;
            trace(params.handle, new_origin, refraction_dir, 0.01f, 1e16f, &fresnelPRD);
            refraction_color = fresnelPRD.accumulated_color;
        }

        // Mix reflection and refraction using Fresnel to enhance transparency
        float3 final_color = (1.0f - transparency_factor) * base_color +
                             transparency_factor * ((make_float3(1.0f) - fresnel) * refraction_color + fresnel * reflection_color);

        //final_color += prd->throughput * emission;
        prd->accumulated_color += clamp(final_color, 0.0f, 1.0f);
        return ;
    }
    //prd->accumulated_color = interpolated_normal;
    //return ;


    // **Simplified Specular Reflection for Non-Glass Materials**

    // Accumulate emission from the current hit
    //prd->accumulated_color += prd->throughput * emission;

    float3 diffuse = base_color * (1.0f - metallic) / M_PI;
    prd->accumulated_color += prd->throughput * diffuse;

    // Calculate Fresnel term
    float fresnel = fresnelSchlick(cosTheta, F0_scalar);

    // Compute reflection direction
    float3 reflection_dir = reflect(optixGetWorldRayDirection(), interpolated_normal);
    reflection_dir = safe_normalize(reflection_dir);

    // **Scale Specular Strength Based on Roughness**
    // Higher roughness results in weaker specular reflections
    float specular_strength = lerp(1.0f, 0.3f, roughness); // Adjust the 0.3f as needed for desired effect
    float3 specular = F0 * fresnel * metallic * specular_strength;

    // Update throughput based on Fresnel and roughness
    float3 new_throughput = prd->throughput * specular;

    // Offset ray origin to avoid self-intersection
    float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    float3 new_origin = safe_normalize(hit_point + 1e-3f * interpolated_normal);

    // Prepare new PerRayData
    PerRayData new_prd;
    new_prd.accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    new_prd.throughput = new_throughput; // Adjusted by Fresnel
    new_prd.depth = prd->depth + 1;
    new_prd.state = prd->state; // Pass the current seed

    // Russian Roulette termination (simplified)
    if (new_prd.depth >= MIN_DEPTH) {
        float probability = fmaxf(new_prd.throughput.x, fmaxf(new_prd.throughput.y, new_prd.throughput.z));
        if (curand_uniform(&new_prd.state) > probability) {
            // Terminate the ray
            return;
        }
        // Otherwise, scale throughput
        new_prd.throughput /= probability;
    }

    // Trace the new reflection ray
    trace(params.handle, new_origin, reflection_dir, 0.01f, 1e16f, &new_prd);

    // Accumulate color from the new ray
    prd->accumulated_color += new_prd.accumulated_color;
}
extern "C" __global__ void __anyhit__ah() {
    //printf("anyhit");
    const HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    // Retrieve barycentric coordinates
    float2 barycentrics = optixGetTriangleBarycentrics();
    float alpha_weight = 1.0f - barycentrics.x - barycentrics.y;
    float beta = barycentrics.x;
    float gamma = barycentrics.y;

    unsigned int primitive_index = optixGetPrimitiveIndex();
    uint3* index_buffer = reinterpret_cast<uint3*>(hit_data->index_buffer);
    uint3 vertex_indices = index_buffer[primitive_index];

    float2* uv_buffer = reinterpret_cast<float2*>(hit_data->uv_buffer);
    float2 uv0 = uv_buffer[vertex_indices.x];
    float2 uv1 = uv_buffer[vertex_indices.y];
    float2 uv2 = uv_buffer[vertex_indices.z];

    // Interpolate UV coordinates using barycentric weights
    float2 uv = alpha_weight * uv0 + beta * uv1 + gamma * uv2;

    // Sample the texture's alpha channel
    float4 tex_color = sampleTexture(hit_data->texture, uv);
    float alpha = tex_color.w;

    // Define a transparency threshold
    const float alpha_threshold = 0.1f; // Adjust this value as needed

    if (alpha < alpha_threshold) {

        // Ignore the intersection, treat it as transparent
        optixIgnoreIntersection();
    }
    // If alpha >= threshold, accept the hit and proceed to the closest-hit shader
}
extern "C" __global__ void __miss__ms() {
    const MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    PerRayData *prd = getPRD();

    float3 background_color = make_float3(miss_data->bg_color.x, miss_data->bg_color.y, miss_data->bg_color.z);

    prd->accumulated_color += prd->throughput * background_color;
}
