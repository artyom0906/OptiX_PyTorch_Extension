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

#include <optix.h>

#include "../optixDynamicGeometry.h"

#include "../../sutil/vec_math.h"
#include "../../cuda/helpers.h"

extern "C" {
    __constant__ Params params;
}

constexpr unsigned int SBT_STRIDE_COLLAPSE = 0;

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    float3*                prd
)
{
    unsigned int p0, p1, p2;
    p0 = __float_as_uint( prd->x );
    p1 = __float_as_uint( prd->y );
    p2 = __float_as_uint( prd->z );
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
        p0, p1, p2 );
    prd->x = __uint_as_float( p0 );
    prd->y = __uint_as_float( p1 );
    prd->z = __uint_as_float( p2 );
}


static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
        __uint_as_float( optixGetPayload_0() ),
        __uint_as_float( optixGetPayload_1() ),
        __uint_as_float( optixGetPayload_2() )
    );
}


extern "C" __global__ void __raygen__rg()
{
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

    const float3 direction = normalize( d.x * U + d.y * V + W );
    float3       payload_rgb = make_float3( 0.5f, 0.5f, 0.5f );

    trace( params.handle,
        eye,
        direction,
        0.00f,  // tmin
        1e16f,  // tmax
        &payload_rgb );

    params.frame_buffer[idx.y * params.width + idx.x] = make_color( payload_rgb );
}


extern "C" __global__ void __miss__ms()
{
    MissData* rt_data = reinterpret_cast< MissData* >( optixGetSbtDataPointer() );
    float3    payload = getPayload();
    setPayload( make_float3( rt_data->bg_color.x, rt_data->bg_color.y, rt_data->bg_color.z ) );
}


extern "C" __global__ void __closesthit__ch()
{
    const HitGroupData* hit_data = reinterpret_cast< HitGroupData* >( optixGetSbtDataPointer() );

    // fetch current triangle vertices
    float3 data[3];
    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
        optixGetRayTime(), data );

    // Retrieve material properties
    float3 color = hit_data->color;     // Material color
    float IOR = hit_data->IOR;         // Index of Refraction
    int is_glass = hit_data->is_glass; // Material type (1 = glass, 0 = other)

    // compute triangle normal
    data[1] -= data[0];
    data[2] -= data[0];
    float3 normal = make_float3(
        data[1].y*data[2].z - data[1].z*data[2].y,
        data[1].z*data[2].x - data[1].x*data[2].z,
        data[1].x*data[2].y - data[1].y*data[2].x );
    const float s = 0.5f / sqrtf( normal.x*normal.x + normal.y*normal.y + normal.z*normal.z );
    color.x = is_glass==1 ? color.x : 0.f;
    // convert normal to color and store in payload (normal*s + make_float3( 0.5 )) *
    setPayload( color );
}
