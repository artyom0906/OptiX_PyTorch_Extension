#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

#include <torch/extension.h>
#include <optix.h>
#include <cuda_runtime.h>

#include "ResourceTypes.h"
#include "DeviceContext.cuh"
#include "GeometryResource.h"
#include "TextureResource.h"
#include "MaterialResource.h"

namespace optix_renderer {

// Central resource management system
class ResourceManager {
public:
    ResourceManager();
    ~ResourceManager();

    // Resource creation
    GeometryHandle createGeometry(
        const torch::Tensor& vertices,
        std::optional<torch::Tensor> indices = std::nullopt,
        std::optional<torch::Tensor> normals = std::nullopt,
        std::optional<torch::Tensor> texCoords = std::nullopt,
        std::optional<torch::Tensor> tangents = std::nullopt,
        std::optional<torch::Tensor> bitangents = std::nullopt
    );
    TextureHandle createTexture(const torch::Tensor& data, TextureType type = TextureType::RGB);
    MaterialHandle createMaterial(MaterialType type);

    // Resource access
    GeometryResource* getGeometry(GeometryHandle handle);
    TextureResource* getTexture(TextureHandle handle);
    MaterialResource* getMaterial(MaterialHandle handle);

    // Device context management
    DeviceContext* getDeviceContext(int deviceId = 0);
    std::vector<int> getAvailableDevices() const;

    // Future extension point for MDL
    void registerMaterialSystem(std::shared_ptr<IMaterialSystem> materialSystem);
    
    // Legacy shader support has been removed
    // We now use a unified parameter passing system

private:
    // Unique ID generator for resources
    ResourceID generateResourceId();

    // Resource maps
    std::unordered_map<ResourceID, std::shared_ptr<GeometryResource>> m_geometries;
    std::unordered_map<ResourceID, std::shared_ptr<TextureResource>> m_textures;
    std::unordered_map<ResourceID, std::shared_ptr<MaterialResource>> m_materials;

    // Device contexts (one per GPU)
    std::vector<std::unique_ptr<DeviceContext>> m_deviceContexts;
    
    // Current resource ID counter
    ResourceID m_nextResourceId;

    // Material system for future MDL support
    std::shared_ptr<IMaterialSystem> m_materialSystem;
    
    // Legacy shader support removed
};

} // namespace optix_renderer