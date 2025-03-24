#include <optix.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>  // For CUtexObject
#include <vector_types.h>
#include <vector_functions.h>
#include <stdio.h>   // For printf in device code
#include "material_types.h"  // Include the enhanced material types
#include <vector_types.h>
#include <vector_functions.h>

using namespace optix_renderer;

// Simple vector math helper functions
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

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(const float3& v) {
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
// Define hit group data structure - must match HitGroupData in LaunchParams.h
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
    CUtexObject albedo_texture;         // Base color texture
    CUtexObject normal_texture;         // Normal map texture
    CUtexObject metallic_roughness_texture;  // Combined metallic/roughness texture
    CUtexObject emission_texture;       // Emission texture
    CUtexObject specular_texture;       // Specular texture
    CUtexObject specular_tint_texture;  // Specular tint texture
    CUtexObject sheen_texture;          // Sheen texture
    CUtexObject clearcoat_texture;      // Clearcoat texture
    
    // Flags for geometry features
    bool has_normals;
    bool has_texcoords;
    bool has_tangents;
    bool has_bitangents;
    
    // Legacy compatibility fields (direct color access)
    float3 albedo;
    float3 emission;
};

// Define the params structure directly in shader 
// This must match LaunchParams in include/LaunchParams.h
struct Params {
    float3* image;          // Raw pointer to output buffer
    float3 camera_pos;
    float3 camera_u;
    float3 camera_v;
    float3 camera_w;
    OptixTraversableHandle traversable;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_pixel;
    unsigned int max_depth;
    CUdeviceptr output_buffer;  // Additional field for direct device ptr access
};
// Compute triangle normal
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

    // Compute geometric normal
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 normal = cross(e1, e2);

    return normalize(normal);
}
// Get texture coordinates at hit point - simplified version
__device__ float2 getTexCoords(const HitGroupData* data, const float2& barycentrics) {
    if (!data->has_texcoords || !data->texcoords) {
        return make_float2(0.5f, 0.5f);  // Default UV in the middle of texture
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
        // Debug info about texture sampling for specific pixel positions
        uint3 launchIndex = optixGetLaunchIndex();
        bool isDebugPixel = ((launchIndex.x % 50 == 0) && (launchIndex.y % 50 == 0)) || 
                            (launchIndex.x == 100 && launchIndex.y == 100);
                            
        if (isDebugPixel) {
            printf("DEBUG: Pixel(%d,%d) - Sampling texture at UV: (%f, %f)\n", 
                   launchIndex.x, launchIndex.y, uv.x, uv.y);
        }
        
        // Use tex2D to sample the texture
        float4 result;
        
        // WARNING: We have to be careful here: if tex is not a valid texture object
        // this could cause a crash. tex objects should be 0 if not valid
        if (tex != 0) {
            result = tex2D<float4>(tex, uv.x, uv.y);
            
            // Debug output for specific pixels
            if (isDebugPixel) {
                printf("DEBUG: Texture color at (%f, %f): (%f, %f, %f, %f)\n", 
                       uv.x, uv.y, result.x, result.y, result.z, result.w);
            }
            
            return result;
        } else {
            // If we got here, tex is 0, which is invalid
            if (isDebugPixel) {
                printf("WARNING: Invalid texture handle (0) passed to sampleTexture\n");
            }
        }
    }
    
    // No valid texture, return a debug color
    return make_float4(1.0f, 0.0f, 1.0f, 1.0f);  // Magenta if texture missing
}

// This params variable will be accessed by the shader through optixLaunch
// The name "params" must match pipelineLaunchParamsVariableName in Renderer.cu
extern "C" {
    __constant__ Params params;
}
// Ray generation shader - no tracing, just outputs a red/black pattern
extern "C" __global__ void __raygen__renderFrame() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    // Debug output - only for thread (0,0)
    //if (idx.x == 0 && idx.y == 0) {
    //    printf("SHADER PARAMS DEBUG (from device params):\n");
    //    printf("  image ptr: %p\n", params.image);
    //    printf("  output_buffer: %llu\n", (unsigned long long)params.output_buffer);
    //    printf("  width: %u, height: %u\n", params.width, params.height);
    //    printf("  camera_pos: (%f, %f, %f)\n",
    //           params.camera_pos.x, params.camera_pos.y, params.camera_pos.z);
    //    printf("  camera_u: (%f, %f, %f)\n",
    //           params.camera_u.x, params.camera_u.y, params.camera_u.z);
    //    printf("  camera_v: (%f, %f, %f)\n",
    //           params.camera_v.x, params.camera_v.y, params.camera_v.z);
    //    printf("  camera_w: (%f, %f, %f)\n",
    //           params.camera_w.x, params.camera_w.y, params.camera_w.z);
    //    printf("  traversable: %llu\n", (unsigned long long)params.traversable);
    //}
    
    // Calculate buffer index
    unsigned int x = idx.x;
    unsigned int y = idx.y;
    
    // Default values if params are invalid
    unsigned int width = dim.x; // Default to launch dimensions
    unsigned int height = dim.y;
    
    // Use params values if they appear valid
    if (params.width > 0) width = params.width;
    if (params.height > 0) height = params.height;
    
    // Ensure we're within bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Create a checkboard pattern
    bool isRed = ((x / 20) + (y / 20)) % 2 == 0;
    
    // Add border to visualize frame boundary
    bool isBorder = (x < 5 || y < 5 || x > width-5 || y > height-5);
    if (isBorder) {
        isRed = !isRed; // Invert border color
    }
    
    // Set default color: red/black checkerboard - but make very bright to debug
    float3 color;
    if (isRed) {
        color = make_float3(1.0f, 0.0f, 1.0f); // Bright orange for checkerboard pattern
    } else {
        color = make_float3(1.0f, 1.0f, 0.0f); // Grey instead of black for checkerboard pattern
    }
    
    // If we have a valid traversable handle, trace a ray to see if it hits the cube
    if (params.traversable) {
        // Trace a ray from camera to this pixel
        const float3 ray_origin = params.camera_pos;
        
        // Simplified direction calculation
        const float2 screen_pos = make_float2(
            (float)x / width * 2.0f - 1.0f,  // Map to [-1, 1]
            (float)y / height * 2.0f - 1.0f  // Map to [-1, 1]
        );
        
        // Calculate ray direction using camera basis
        float3 dir = make_float3(
            params.camera_u.x * screen_pos.x + params.camera_v.x * screen_pos.y + params.camera_w.x,
            params.camera_u.y * screen_pos.x + params.camera_v.y * screen_pos.y + params.camera_w.y,
            params.camera_u.z * screen_pos.x + params.camera_v.z * screen_pos.y + params.camera_w.z
        );
        
        // Normalize the direction
        float length = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
        const float3 ray_direction = make_float3(dir.x/length, dir.y/length, dir.z/length);
        
        // Extensive debug output from pixel centers and corners
        //if ((x == width/2 && y == height/2) ||
        //    (x == 0 && y == 0) ||
        //    (x == width-1 && y == height-1) ||
        //    (x == 0 && y == height-1) ||
        //    (x == width-1 && y == 0)) {
        //    printf("DEBUG RAY: Pixel(%d,%d) Origin:(%f,%f,%f) Dir:(%f,%f,%f)\n",
        //           x, y,
        //           ray_origin.x, ray_origin.y, ray_origin.z,
        //           ray_direction.x, ray_direction.y, ray_direction.z);
        //}
        
        // Trace the ray - use only standard flags
        // Initialize ray payload variables
        unsigned int p0 = 0;  // Red
        unsigned int p1 = 0;  // Green
        unsigned int p2 = 0;  // Blue
        // We don't need to manually specify instance SBT offset
        // OptiX will automatically use the instanceId to select the right hit shader
        unsigned int sbtOffset = 0;

        // Always debug the SBT setup for the center ray
        //if (idx.x == width/2 && idx.y == height/2) {
        //    printf("CENTER RAY: direction = (%f, %f, %f)\n",
        //           ray_direction.x, ray_direction.y, ray_direction.z);
        //    printf("SBT setup: hitgroupCount = 2, stride = %lu bytes\n",
        //           sizeof(HitGroupData) + OPTIX_SBT_RECORD_HEADER_SIZE);
        //}

        optixTrace(
            params.traversable,
            ray_origin,
            ray_direction,
            0.0f,                  // Min distance
            1e16f,                 // Max distance
            0.0f,                  // Ray time
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,     // We just want to know if we hit
            sbtOffset,             // SBT offset - use 0 for cube, 1 for floor
            1,                     // SBT stride - use proper stride if multiple hit groups are used
            0,                     // Miss SBT index
            p0, p1, p2             // Payload (RGB values)
        );

        // Convert payload to float RGB values
        color.x = __uint_as_float(p0);
        color.y = __uint_as_float(p1);
        color.z = __uint_as_float(p2);


        
        // Log all hits to better understand what's being hit
        //if (hit) {
        //    // If the ray hit the cube, color it green
        //    color = make_float3(0.0f, 1.0f, 0.0f); // Green for cube hit
        //
        //    // Debug output when a hit is found (log every 20th pixel to avoid flooding the output)
        //    if (x % 20 == 0 && y % 20 == 0) {
        //        printf("HIT: Pixel(%d,%d) Origin:(%f,%f,%f) Dir:(%f,%f,%f)\n",
        //            x, y,
        //            ray_origin.x, ray_origin.y, ray_origin.z,
        //            ray_direction.x, ray_direction.y, ray_direction.z);
        //    }
        //}
    }
    
    // Debug output for specific pixel
    //if (x == 300 && y == 300) {
    //    printf("DEBUG: Final color at (300,300): (%f, %f, %f)\n", color.x, color.y, color.z);
    //}
    
    // Output to buffer if params.output_buffer is valid
    float3* output = (float3*)params.image;
    if (params.output_buffer != 0) {

        unsigned int idx = y * width + x;
        if (x < width && y < height) {
            output[idx] = color;
        }
    }
}

// Miss shader - minimal implementation
extern "C" __global__ void __miss__environment() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    // Calculate buffer index
    unsigned int x = idx.x;
    unsigned int y = idx.y;

    // Default values if params are invalid
    unsigned int width = dim.x; // Default to launch dimensions
    unsigned int height = dim.y;

    // Use params values if they appear valid
    if (params.width > 0) width = params.width;
    if (params.height > 0) height = params.height;

    // Ensure we're within bounds
    if (x >= width || y >= height) {
        return;
    }

    // Create a checkboard pattern
    bool isRed = ((x / 20) + (y / 20)) % 2 == 0;

    // Add border to visualize frame boundary
    bool isBorder = (x < 5 || y < 5 || x > width-5 || y > height-5);
    if (isBorder) {
        isRed = !isRed; // Invert border color
    }

    // Set default color: red/black checkerboard - but make very bright to debug
    float3 color;
    if (isRed) {
        color = make_float3(0.0f, 0.0f, 0.0f); // Bright orange for checkerboard pattern
    } else {
        color = make_float3(1.0f, 0.3f, 0.3f); // Grey instead of black for checkerboard pattern
    }
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

// Closest hit shader - gets color from the hit group SBT record
extern "C" __global__ void __closesthit__radiance() {
    // Get the SBT record data to access the color for this object
    const HitGroupData* data = (const HitGroupData*)optixGetSbtDataPointer();
    
    // Get hit information
    const float2 barycentrics = optixGetTriangleBarycentrics();

    // Get texture coordinates
    float2 texCoords = getTexCoords(data, barycentrics);

    // Default color is the material albedo
    float3 color = data->albedo;

    // Check if we have texcoords - directly visualize them if available
    if (data->has_texcoords) {
        // Just show the texture coordinates as colors for debugging
        // Red = U coordinate, Green = V coordinate
        color = make_float3(texCoords.x, texCoords.y, 0.0f);
    }
    
    // Test if we have a texture handle
    CUtexObject texHandle = data->albedo_texture;
    
    // Debug texture handle for specific pixels
    uint3 launchIndex = optixGetLaunchIndex();
    bool isDebugPixel = ((launchIndex.x % 100 == 0) && (launchIndex.y % 100 == 0)) || 
                         (launchIndex.x == 100 && launchIndex.y == 100);
    
    if (isDebugPixel) {
        printf("DEBUG: Pixel(%d,%d) - Texture handle: %llu, Has texcoords: %s\n", 
               launchIndex.x, launchIndex.y, 
               (unsigned long long)texHandle,
               data->has_texcoords ? "true" : "false");
    }
    
    if (texHandle) {
        // Now try to sample the texture
        float4 texColor = sampleTexture(texHandle, texCoords);
        
        // Use a mix of visualization techniques:
        
        // 1. Show actual texture in most areas
        color = make_float3(texColor.x, texColor.y, texColor.z);
        
        // 2. Add grid lines for reference
        float gridSize = 0.25f;  // Larger grid cells - 4x4 grid over texture
        bool onGridX = fmodf(texCoords.x, gridSize) < 0.01f || fmodf(texCoords.x, gridSize) > (gridSize - 0.01f);
        bool onGridY = fmodf(texCoords.y, gridSize) < 0.01f || fmodf(texCoords.y, gridSize) > (gridSize - 0.01f);
        
        if (onGridX || onGridY) {
            // Add semi-transparent black grid lines
            color = make_float3(color.x * 0.3f, color.y * 0.3f, color.z * 0.3f);
        }
        
        // 3. Show UV borders more clearly
        float uvBorder = 0.05f; // Wider border for clarity
        if (texCoords.x < uvBorder || texCoords.x > (1.0f - uvBorder) ||
            texCoords.y < uvBorder || texCoords.y > (1.0f - uvBorder)) {
            // White borders at UV edges
            color = make_float3(1.0f, 1.0f, 1.0f);
        }
        
        // 4. Show corner markers to diagnose texture orientation and flipping
        float cornerSize = 0.1f;
        
        // Top-left corner: Red marker
        if (texCoords.x < cornerSize && texCoords.y > (1.0f - cornerSize)) {
            float mix = 0.7f;  // How much to blend with texture
            color = make_float3(1.0f, 0.0f, 0.0f) * mix + color * (1.0f - mix);
        }
        
        // Top-right corner: Green marker
        if (texCoords.x > (1.0f - cornerSize) && texCoords.y > (1.0f - cornerSize)) {
            float mix = 0.7f;
            color = make_float3(0.0f, 1.0f, 0.0f) * mix + color * (1.0f - mix);
        }
        
        // Bottom-left corner: Blue marker
        if (texCoords.x < cornerSize && texCoords.y < cornerSize) {
            float mix = 0.7f;
            color = make_float3(0.0f, 0.0f, 1.0f) * mix + color * (1.0f - mix);
        }
        
        // Bottom-right corner: Yellow marker
        if (texCoords.x > (1.0f - cornerSize) && texCoords.y < cornerSize) {
            float mix = 0.7f;
            color = make_float3(1.0f, 1.0f, 0.0f) * mix + color * (1.0f - mix);
        }
    } else {
        // No texture - show a pattern to indicate no texture was found
        color = make_float3(0.5f, 0.1f, 0.1f);  // Dark red means no texture
    }

    //else {
    // If no texture, use simple shading with normal
        //float3 normal = computeNormal(data, barycentrics);
        //float ndotl = max(0.0f, normal.y * 0.5f + 0.5f); // Simple light from ab

        // Mix original color with lighting
        //color = color * ndotl;

        // Add UV coordinate visualization if we have texture coordinates
        //if (data->has_texcoords) {
        //    float uvBorder = 0.02f; // Border size
        //    if (texCoords.x < uvBorder || texCoords.x > 1.0f - uvBorder ||
        //        texCoords.y < uvBorder || texCoords.y > 1.0f - uvBorder) {
        //        // Show UV coordinates along edges
        //        color = make_float3(texCoords.x, texCoords.y, 0.0f);
        //    }
        //}
	//}
    
    // Debug output
    //if (optixGetLaunchIndex().x == 300 && optixGetLaunchIndex().y == 300) {
    //    printf("HIT: SBT index %u, albedo: (%f, %f, %f)\n",
    //           optixGetSbtGASIndex(), data->albedo.x, data->albedo.y, data->albedo.z);
    //}
    
    // Set the payload values to the SBT record's color
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}