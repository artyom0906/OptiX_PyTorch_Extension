#pragma once

#include <vector_types.h>
#include <stdint.h> // For uint32_t

namespace optix_renderer {

// Material types enum - renamed to avoid conflict with ResourceTypes.h
enum OptiXMaterialType {
    OPTIX_LAMBERTIAN = 0,  // Simple diffuse material
    OPTIX_PBR,             // Physically-based renderer material
    OPTIX_GLASS,           // Transparent glass material
    OPTIX_EMISSIVE,        // Light-emitting material
    OPTIX_MIRROR,          // Perfect mirror reflections
    OPTIX_MDL              // Material Definition Language (MDL) material
};

// Enhanced material structure for the OptiX shader - Phase 1 implementation
struct SimpleMaterial {
    float3 albedo;          // Base color for all material types
    float3 emission;        // Emission color and strength (if emissive)
    float metallic;         // 0 = dielectric, 1 = metal
    float roughness;        // 0 = smooth, 1 = rough
    float transmission;     // 0 = opaque, 1 = transparent
    float ior;              // Index of refraction for transparent materials
    float specular;         // Specular reflection intensity
    float specularTint;     // Tint for specular highlights
    float sheen;            // Cloth-like sheen effect
    float sheenTint;        // Tint for sheen effect
    float clearcoat;        // Clear coat layer intensity
    float clearcoatGloss;   // Glossiness of clearcoat layer
    float anisotropic;      // Directional scattering
    
    // Texture indices (-1 if not used)
    int albedoTexture;      // Base color texture
    int normalTexture;      // Normal map texture
    int metallicRoughnessTexture; // Combined metallic/roughness texture (r=metallic, g=roughness)
    int emissionTexture;    // Emission map texture
    int specularTexture;    // Specular texture
    int specularTintTexture; // Specular tint texture
    
    // Material classification
    int materialType;       // Corresponds to OptiXMaterialType enum
    
    // MDL support - allows binding to external MDL material instances
    int mdlMaterialId;      // ID for MDL material (-1 if not MDL)
    
    // Material flags for optimized execution paths
    uint32_t flags;         // Bit flags for material properties (has textures, etc.)
};

// Flag definitions for material properties
enum MaterialFlags {
    MATERIAL_HAS_ALBEDO_TEXTURE = (1 << 0),
    MATERIAL_HAS_NORMAL_TEXTURE = (1 << 1),
    MATERIAL_HAS_METALLIC_ROUGHNESS_TEXTURE = (1 << 2),
    MATERIAL_HAS_EMISSION_TEXTURE = (1 << 3),
    MATERIAL_IS_EMISSIVE = (1 << 4),
    MATERIAL_IS_TRANSPARENT = (1 << 5),
    MATERIAL_HAS_ANISOTROPY = (1 << 6),
    MATERIAL_HAS_CLEARCOAT = (1 << 7),
    MATERIAL_HAS_SHEEN = (1 << 8),
    MATERIAL_IS_MDL = (1 << 9)
};

// Utility functions for working with material flags
#ifdef __CUDACC__
__host__ __device__
#endif
inline void setMaterialFlag(uint32_t& flags, MaterialFlags flag, bool value) {
    if (value)
        flags |= flag;
    else
        flags &= ~flag;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline bool getMaterialFlag(uint32_t flags, MaterialFlags flag) {
    return (flags & flag) != 0;
}

} // namespace optix_renderer