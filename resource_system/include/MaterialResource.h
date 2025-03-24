#pragma once

#include <unordered_map>
#include <string>
#include "Resource.h"

namespace optix_renderer {

// Holds material parameters and textures
class MaterialResource : public Resource {
public:
    MaterialResource(ResourceID id, MaterialType type)
        : Resource(id, ResourceType::MATERIAL)
        , m_materialType(type)
    {
        // Initialize with default parameters based on material type
        initializeDefaults();
    }
    
    // Material type access
    MaterialType getMaterialType() const { return m_materialType; }
    
    // Parameter management
    void setParameter(const std::string& name, const MaterialParameter& value) {
        m_parameters[name] = value;
    }
    
    MaterialParameter getParameter(const std::string& name) const {
        auto it = m_parameters.find(name);
        if (it != m_parameters.end()) {
            return it->second;
        }
        return MaterialParameter(); // Return default parameter
    }
    
    bool hasParameter(const std::string& name) const {
        return m_parameters.find(name) != m_parameters.end();
    }
    
    // Texture binding
    void setTexture(const std::string& name, TextureHandle textureHandle) {
        m_textures[name] = textureHandle;
    }
    
    TextureHandle getTexture(const std::string& name) const {
        auto it = m_textures.find(name);
        if (it != m_textures.end()) {
            return it->second;
        }
        return 0; // Invalid handle
    }
    
    bool hasTexture(const std::string& name) const {
        return m_textures.find(name) != m_textures.end();
    }
    
    // Access to all parameters/textures
    const std::unordered_map<std::string, MaterialParameter>& getAllParameters() const {
        return m_parameters;
    }
    
    const std::unordered_map<std::string, TextureHandle>& getAllTextures() const {
        return m_textures;
    }
    
private:
    void initializeDefaults() {
        // Set default parameters based on material type
        switch (m_materialType) {
            case MaterialType::LAMBERTIAN:
                setParameter("albedo", MaterialParameter(new float[3]{0.8f, 0.8f, 0.8f}));
                break;
                
            case MaterialType::PBR:
                setParameter("base_color", MaterialParameter(new float[3]{0.8f, 0.8f, 0.8f}));
                setParameter("metallic", MaterialParameter(0.0f));
                setParameter("roughness", MaterialParameter(0.5f));
                setParameter("specular", MaterialParameter(0.5f));
                break;
                
            case MaterialType::GLASS:
                setParameter("transmission", MaterialParameter(1.0f));
                setParameter("ior", MaterialParameter(1.5f));
                setParameter("roughness", MaterialParameter(0.0f));
                break;
                
            case MaterialType::EMISSIVE:
                setParameter("emission_color", MaterialParameter(new float[3]{1.0f, 0.9f, 0.7f}));
                setParameter("emission_strength", MaterialParameter(1.0f));
                break;
                
            case MaterialType::MIRROR:
                setParameter("reflectivity", MaterialParameter(1.0f));
                setParameter("tint", MaterialParameter(new float[3]{1.0f, 1.0f, 1.0f}));
                break;
        }
    }

    MaterialType m_materialType;
    std::unordered_map<std::string, MaterialParameter> m_parameters;
    std::unordered_map<std::string, TextureHandle> m_textures;
};

} // namespace optix_renderer