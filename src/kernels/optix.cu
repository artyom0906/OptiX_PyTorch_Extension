/*

* SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
                                                        * SPDX-License-Identifier: BSD-3-Clause
                                                 *
                                                         * Redistribution and use in source and binary forms, with or without
                        * modification, are permitted provided that the following conditions are met:
    *
    * 1. Redistributions of source code must retain the above copyright notice, this
                    * list of conditions and the following disclaimer.
                            *
            * 2. Redistributions in binary form must reproduce the above copyright notice,
    * this list of conditions and the following disclaimer in the documentation
                            * and/or other materials provided with the distribution.
                               *
                       * 3. Neither the name of the copyright holder nor the names of its
                                 * contributors may be used to endorse or promote products derived from
                               * this software without specific prior written permission.
                                       *
                                               * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
                       * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
            * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
                    * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
                            * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
            * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
                                                            * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
                    * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
                    */

#include "../common.h"

            extern "C" {
    __constant__ Params params;
}

//------------------------------------------------------------------------------
//
// Orthonormal basis helper
//
//------------------------------------------------------------------------------


struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if( fabs(m_normal.x) > fabs(m_normal.z) )
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y =  m_normal.x;
            m_binormal.z =  0;
        }
        else
        {
            m_binormal.x =  0;
            m_binormal.y = -m_normal.z;
            m_binormal.z =  m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross( m_binormal, m_normal );
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

//------------------------------------------------------------------------------
//
// Utility functions
//
//------------------------------------------------------------------------------


static __forceinline__ __device__ RadiancePRD loadClosesthitRadiancePRD()
{
    RadiancePRD prd = {};

    prd.attenuation.x = __uint_as_float( optixGetPayload_0() );
    prd.attenuation.y = __uint_as_float( optixGetPayload_1() );
    prd.attenuation.z = __uint_as_float( optixGetPayload_2() );
    prd.seed  = optixGetPayload_3();
    prd.depth = optixGetPayload_4();
     prd.inside = optixGetPayload_18();
    return prd;
}

static __forceinline__ __device__ RadiancePRD loadMissRadiancePRD()
{
    RadiancePRD prd = {};
    return prd;
}

static __forceinline__ __device__ void storeClosesthitRadiancePRD( RadiancePRD prd )
{
    optixSetPayload_0( __float_as_uint( prd.attenuation.x ) );
    optixSetPayload_1( __float_as_uint( prd.attenuation.y ) );
    optixSetPayload_2( __float_as_uint( prd.attenuation.z ) );

    optixSetPayload_3( prd.seed );
    optixSetPayload_4( prd.depth );

    optixSetPayload_5( __float_as_uint( prd.emitted.x ) );
    optixSetPayload_6( __float_as_uint( prd.emitted.y ) );
    optixSetPayload_7( __float_as_uint( prd.emitted.z ) );

    optixSetPayload_8( __float_as_uint( prd.radiance.x ) );
    optixSetPayload_9( __float_as_uint( prd.radiance.y ) );
    optixSetPayload_10( __float_as_uint( prd.radiance.z ) );

    optixSetPayload_11( __float_as_uint( prd.origin.x ) );
    optixSetPayload_12( __float_as_uint( prd.origin.y ) );
    optixSetPayload_13( __float_as_uint( prd.origin.z ) );

    optixSetPayload_14( __float_as_uint( prd.direction.x ) );
    optixSetPayload_15( __float_as_uint( prd.direction.y ) );
    optixSetPayload_16( __float_as_uint( prd.direction.z ) );

    optixSetPayload_17( prd.done );
    optixSetPayload_18( prd.inside );
}


static __forceinline__ __device__ void storeMissRadiancePRD( RadiancePRD prd )
{
    optixSetPayload_5( __float_as_uint( prd.emitted.x ) );
    optixSetPayload_6( __float_as_uint( prd.emitted.y ) );
    optixSetPayload_7( __float_as_uint( prd.emitted.z ) );

    optixSetPayload_8( __float_as_uint( prd.radiance.x ) );
    optixSetPayload_9( __float_as_uint( prd.radiance.y ) );
    optixSetPayload_10( __float_as_uint( prd.radiance.z ) );

    optixSetPayload_17( prd.done );
}


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r   = sqrtf( u1 );
    const float phi = 2.0f*M_PIf * u2;
    p.x = r * cosf( phi );
    p.y = r * sinf( phi );

    // Project up to hemisphere.
    p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}


static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        RadiancePRD&           prd
)
{
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18;

    u0 = __float_as_uint( prd.attenuation.x );
    u1 = __float_as_uint( prd.attenuation.y );
    u2 = __float_as_uint( prd.attenuation.z );
    u3 = prd.seed;
    u4 = prd.depth;
    u18 = prd.inside;

    if(isnan(ray_origin.x) || isnan(ray_origin.y) || isnan(ray_origin.z) ||
        isnan(ray_direction.x) || isnan(ray_direction.y) || isnan(ray_direction.z)){
        prd.done = true;
        return;
    }

    // Note:
    // This demonstrates the usage of the OptiX shader execution reordering
    // (SER) API.  In the case of this computationally simple shading code,
    // there is no real performance benefit.  However, with more complex shaders
    // the potential performance gains offered by reordering are significant.
    optixTraverse(
            PAYLOAD_TYPE_RADIANCE,
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            0,                        // missSBTIndex
            u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18 );
    optixReorder(
            // Application specific coherence hints could be passed in here
    );

    optixInvoke( PAYLOAD_TYPE_RADIANCE,
                u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18 );

    prd.attenuation = make_float3( __uint_as_float( u0 ), __uint_as_float( u1 ), __uint_as_float( u2 ) );
    prd.seed  = u3;
    prd.depth = u4;

    prd.emitted   = make_float3( __uint_as_float( u5 ), __uint_as_float( u6 ), __uint_as_float( u7 ) );
    prd.radiance  = make_float3( __uint_as_float( u8 ), __uint_as_float( u9 ), __uint_as_float( u10 ) );
    prd.origin    = make_float3( __uint_as_float( u11 ), __uint_as_float( u12 ), __uint_as_float( u13 ) );
    prd.direction = make_float3( __uint_as_float( u14 ), __uint_as_float( u15 ), __uint_as_float( u16 ) );
    prd.done = u17;
    prd.inside = u18;
}


// Returns true if ray is occluded, else false
static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
)
{
    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax, 0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,                         // SBT offset
            RAY_TYPE_COUNT,            // SBT stride
            0                          // missSBTIndex
    );
    return optixHitObjectIsHit();
}
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
//------------------------------------------------------------------------------
//
// Programs
//
//------------------------------------------------------------------------------

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
        prd.inside       = 0;

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


extern "C" __global__ void __miss__radiance()
{
    optixSetPayloadTypes( PAYLOAD_TYPE_RADIANCE );

    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD prd = loadMissRadiancePRD();

    prd.radiance  = make_float3( rt_data->bg_color );
    prd.emitted   = make_float3( 0.f );
    prd.done      = true;

    storeMissRadiancePRD( prd );
}


extern "C" __global__ void __closesthit__radiance()
{
    optixSetPayloadTypes( PAYLOAD_TYPE_RADIANCE );

    HitGroupData* hit_data = (HitGroupData*)optixGetSbtDataPointer();
    RadiancePRD prd = loadClosesthitRadiancePRD();

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
    float3 normalTexS = make_float3(0);
    if(hit_data->normal_map != 0){
        float4 texColor  = sampleTexture(hit_data->normal_map, uv);
        normalTexS = make_float3(texColor.x, texColor.y, texColor.z);

        float3 tangent_space_normal  = make_float3(
                texColor.x * 2.0f - 1.0f,
                texColor.y * 2.0f - 1.0f,
                texColor.z * 2.0f - 1.0f
        );

        // Build TBN matrix
        //float3 T = normalize(interpolated_normal.x > 0.9f ? make_float3(0.0f, 1.0f, 0.0f) : cross(make_float3(1.0f, 0.0f, 0.0f), interpolated_normal));
        //float3 B = cross(interpolated_normal, T);

        //// Transform to world space
        //interpolated_normal = normalize(T * tangent_space_normal.x + B * tangent_space_normal.y + interpolated_normal * tangent_space_normal.z);

        float3 T = normalize(interpolated_tangent);
        float3 B = normalize(interpolated_bitangent);
        float3 N = normalize(interpolated_normal);
        // Transform the tangent space normal to world space
        float3 perturbed_normal = normalize(
                T * tangent_space_normal.x +
                B * tangent_space_normal.y +
                N * tangent_space_normal.z
        );

        // Transform to world space
        interpolated_normal = -perturbed_normal;
    }

    float3 base_color = make_float3(1.0f, 1.0f, 1.0f);
    float metallic = 0.0f;
    float roughness = 1.0f;
    float3 emission = make_float3(0.1f);
    if (hit_data->texture != 0) {
        float4 texture_color_with_alpha = sampleTexture(hit_data->texture, uv);
        base_color  = make_float3(texture_color_with_alpha.x, texture_color_with_alpha.y, texture_color_with_alpha.z);
    }
    if(hit_data->metallic_roughness != 0){
        float4 tex = sampleTexture(hit_data->metallic_roughness, uv);
        metallic = tex.x;
        roughness = tex.y;
    }
    if(hit_data->emission_texture != 0){
        float4 tex = sampleTexture(hit_data->emission_texture, uv);
        emission = make_float3(tex.x*hit_data->emission.x, tex.y*hit_data->emission.y, tex.z*hit_data->emission.z);
    }
    const float3 ray_dir         = optixGetWorldRayDirection();
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    if( prd.depth == 0 )
        prd.emitted = emission;
    else
        prd.emitted = make_float3( 0.0f );

    if (length(emission) > 0.0f) {
        prd.radiance += emission;
    }
    if (hit_data->is_glass) {
        float3 I = -ray_dir; // Incident direction
        float3 N = interpolated_normal; // Surface normal


        // Refractive index (glass ~1.5, air ~1.0)
        float eta = prd.inside ? (1.5f / 1.0f) : (1.0f / 1.5f);

        float cosI = dot(I, N);
        float sinT2 = eta * eta * (1.0f - cosI * cosI);

        if (sinT2 > 1.0f) {
            // Total internal reflection
            prd.direction = reflect(I, N);
        } else {
            float cosT = sqrtf(1.0f - sinT2);
            float3 refraction_dir = normalize(eta * I - (eta * cosI + cosT) * N);
            float3 reflection_dir = reflect(I, N);

            // Compute Fresnel reflectance
            float R0 = (1.0f - eta) / (1.0f + eta);
            R0 = R0 * R0;
            float reflectance = R0 + (1.0f - R0) * powf(1.0f - fabsf(cosI), 5.0f);

            // Randomly choose reflection or refraction
            if (rnd(prd.seed) < reflectance) {
                prd.direction = reflection_dir;
            } else {
                prd.direction = refraction_dir;
                prd.inside = !prd.inside;
            }
        }

        prd.origin = P;
        prd.attenuation *= normalize(base_color*2); // Glass absorbs light as it transmits
        prd.done = false;
        storeClosesthitRadiancePRD(prd);
        return;
    }


    unsigned int seed = prd.seed;
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_sample_hemisphere( z1, z2, w_in );
        Onb onb(interpolated_normal);
        onb.inverse_transform( w_in );
        prd.direction = w_in;
        prd.origin    = P;

        prd.attenuation *= base_color;
    }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;

    prd.done     = false;

    storeClosesthitRadiancePRD( prd );
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
