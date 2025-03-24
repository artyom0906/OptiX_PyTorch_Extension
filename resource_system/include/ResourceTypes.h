#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <vector_types.h> // Include vector_types.h for CUDA vector types
#include <cuda_runtime.h> // For make_float2, make_float3, etc.

namespace optix_renderer {

// Basic type definitions
using ResourceID = uint64_t;
using GeometryHandle = ResourceID;
using TextureHandle = ResourceID;
using MaterialHandle = ResourceID;

// Resource type enumeration
enum class ResourceType {
    NONE,
    GEOMETRY,
    TEXTURE,
    MATERIAL
};

// Texture types
enum class TextureType {
    RGB,        // 3-channel RGB
    RGBA,       // 4-channel RGBA
    NORMAL_MAP, // Normal map (RGB)
    GRAYSCALE,  // Single channel
    HDR         // High dynamic range
};

// Material types
enum class MaterialType {
    LAMBERTIAN,  // Simple diffuse material
    PBR,         // Physically-based renderer material
    GLASS,       // Transparent glass material
    EMISSIVE,    // Light-emitting material
    MIRROR       // Perfect mirror reflections
};

// Parameter types for materials
enum class ParameterType {
    FLOAT,
    FLOAT2,
    FLOAT3,
    FLOAT4,
    INT,
    BOOL,
    TEXTURE,
    COLOR
};

// A generic parameter value that can be any of the supported types
class MaterialParameter {
public:
    // Constructors for different types
    MaterialParameter() : m_type(ParameterType::FLOAT), m_floatValue(0.0f) {}
    
    explicit MaterialParameter(float v) : m_type(ParameterType::FLOAT), m_floatValue(v) {}
    explicit MaterialParameter(const float2 v) : m_type(ParameterType::FLOAT2) { m_float2Value = v;}
    explicit MaterialParameter(const float3 v) : m_type(ParameterType::FLOAT3) { m_float3Value = v;}
    explicit MaterialParameter(const float4 v) : m_type(ParameterType::FLOAT4) { m_float4Value = v;}
    explicit MaterialParameter(int v) : m_type(ParameterType::INT), m_intValue(v) {}
    explicit MaterialParameter(bool v) : m_type(ParameterType::BOOL), m_boolValue(v) {}
    explicit MaterialParameter(TextureHandle v) : m_type(ParameterType::TEXTURE), m_textureValue(v) {}
    
    // Type-checked getters
    ParameterType getType() const { return m_type; }
    float asFloat() const { return (m_type == ParameterType::FLOAT) ? m_floatValue : 0.0f; }
    const float2 asFloat2() const { return (m_type == ParameterType::FLOAT2) ? m_float2Value : make_float2(0.0, 0.0); }
    const float3 asFloat3() const { return (m_type == ParameterType::FLOAT3) ? m_float3Value : make_float3(0.0, 0.0, 0.0); }
    const float4 asFloat4() const { return (m_type == ParameterType::FLOAT4) ? m_float4Value : make_float4(0.0, 0.0, 0.0, 0.0); }
    int asInt() const { return (m_type == ParameterType::INT) ? m_intValue : 0; }
    bool asBool() const { return (m_type == ParameterType::BOOL) ? m_boolValue : false; }
    TextureHandle asTexture() const { return (m_type == ParameterType::TEXTURE) ? m_textureValue : 0; }

private:
    ParameterType m_type;
    
    union {
        float m_floatValue;
        float2 m_float2Value;
        float3 m_float3Value;
        float4 m_float4Value;
        int m_intValue;
        bool m_boolValue;
        TextureHandle m_textureValue;
    };
};

// Material parameter information structure for reflection
struct MaterialParameterInfo {
    std::string name;
    std::string displayName;
    ParameterType type;
    MaterialParameter defaultValue;
    float minValue;  // For numeric types
    float maxValue;  // For numeric types
};

// Forward declarations for material system interface
class IMaterialSystem {
public:
    virtual ~IMaterialSystem() = default;
    
    // Material loading/compilation
    virtual MaterialHandle loadMaterial(const std::string& path, const std::string& name) = 0;
    virtual bool compileMaterial(MaterialHandle handle, int deviceId) = 0;
    
    // Parameter handling
    virtual std::vector<MaterialParameterInfo> getMaterialParameters(MaterialHandle handle) = 0;
    virtual void setMaterialParameter(MaterialHandle handle, const std::string& name, const MaterialParameter& value) = 0;
};

} // namespace optix_renderer