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

struct Params
{
    int                    subframe_index;
    float4*                accum_buffer;
    uchar4*                frame_buffer;
    unsigned int samples_per_launch;
    unsigned int           width;
    unsigned int           height;
    float3                 eye, U, V, W;


    OptixTraversableHandle handle;
};

struct RayGenData
{
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    float3 color;
    float IOR;        // Index of Refraction
    int is_glass;     // 1 if glass, 0 otherwise
    cudaTextureObject_t texture;
    cudaTextureObject_t normal_map;         // Pointer to UV buffer
    cudaTextureObject_t emission_texture; // Emissive texture
    cudaTextureObject_t metallic_roughness; // Metallic and roughness texture
    CUdeviceptr texture_ptr;         // Pointer to UV buffer
    CUdeviceptr index_buffer;       // Pointer to the index buffer
    CUdeviceptr uv_buffer;         // Pointer to UV buffer
    CUdeviceptr normals;         // Pointer to UV buffer
    CUdeviceptr tangents;
    CUdeviceptr bitangents;
};