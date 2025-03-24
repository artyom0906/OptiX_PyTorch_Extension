#include "../include/ResourceManager.cuh"
#include "../include/GeometryResource.h"
#include "../include/TextureResource.h"
#include "../include/MaterialResource.h"

#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <fstream>
#include <sstream>

namespace optix_renderer {

ResourceManager::ResourceManager()
    : m_nextResourceId(1) // Start from 1, 0 is reserved as invalid
{
    // Get the number of CUDA devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    // Create device contexts for all available GPUs
    for (int i = 0; i < deviceCount; ++i) {
        m_deviceContexts.push_back(std::make_unique<DeviceContext>(i));
        m_deviceContexts[i]->initialize();
    }
}

ResourceManager::~ResourceManager()
{
    // Cleanup all device contexts
    for (auto& context : m_deviceContexts) {
        context->destroy();
    }
    m_deviceContexts.clear();
    
    // Clear all resources
    m_geometries.clear();
    m_textures.clear();
    m_materials.clear();
}

GeometryHandle ResourceManager::createGeometry(
    const torch::Tensor& vertices,
    std::optional<torch::Tensor> indices,
    std::optional<torch::Tensor> normals,
    std::optional<torch::Tensor> texCoords,
    std::optional<torch::Tensor> tangents,
    std::optional<torch::Tensor> bitangents)
{
    ResourceID id = generateResourceId();
    
    // Check if provided optional tensors are defined and non-empty
    if (indices.has_value() && (!indices->defined() || indices->numel() == 0)) {
        indices = std::nullopt;
    }
    if (normals.has_value() && (!normals->defined() || normals->numel() == 0)) {
        normals = std::nullopt;
    }
    if (texCoords.has_value() && (!texCoords->defined() || texCoords->numel() == 0)) {
        texCoords = std::nullopt;
    }
    if (tangents.has_value() && (!tangents->defined() || tangents->numel() == 0)) {
        tangents = std::nullopt;
    }
    if (bitangents.has_value() && (!bitangents->defined() || bitangents->numel() == 0)) {
        bitangents = std::nullopt;
    }
    
    // Create geometry resource 
    auto geometry = std::make_shared<GeometryResource>(id, vertices, indices);
    
    // Set additional attributes if provided
    if (normals.has_value()) {
        geometry->setNormals(normals.value());
    }
    if (texCoords.has_value()) {
        geometry->setTexCoords(texCoords.value());
    }
    if (tangents.has_value()) {
        geometry->setTangents(tangents.value());
    }
    if (bitangents.has_value()) {
        geometry->setBitangents(bitangents.value());
    }
    
    m_geometries[id] = geometry;
    
    // Allocate geometry on all devices
    for (auto& context : m_deviceContexts) {
        context->allocateGeometry(id, geometry.get());
    }
    
    return id;
}

TextureHandle ResourceManager::createTexture(const torch::Tensor& data, TextureType type)
{
    ResourceID id = generateResourceId();
    auto texture = std::make_shared<TextureResource>(id, data, type);
    m_textures[id] = texture;
    
    // Allocate texture on all devices
    for (auto& context : m_deviceContexts) {
        context->allocateTexture(id, texture.get());
    }
    
    return id;
}

MaterialHandle ResourceManager::createMaterial(MaterialType type)
{
    ResourceID id = generateResourceId();
    auto material = std::make_shared<MaterialResource>(id, type);
    m_materials[id] = material;
    
    // Allocate material on all devices
    for (auto& context : m_deviceContexts) {
        context->allocateMaterial(id, material.get());
    }
    
    return id;
}

GeometryResource* ResourceManager::getGeometry(GeometryHandle handle)
{
    auto it = m_geometries.find(handle);
    if (it != m_geometries.end()) {
        return it->second.get();
    }
    return nullptr;
}

TextureResource* ResourceManager::getTexture(TextureHandle handle)
{
    auto it = m_textures.find(handle);
    if (it != m_textures.end()) {
        return it->second.get();
    }
    return nullptr;
}

MaterialResource* ResourceManager::getMaterial(MaterialHandle handle)
{
    auto it = m_materials.find(handle);
    if (it != m_materials.end()) {
        return it->second.get();
    }
    return nullptr;
}

DeviceContext* ResourceManager::getDeviceContext(int deviceId)
{
    for (auto& context : m_deviceContexts) {
        if (context->getDeviceId() == deviceId) {
            return context.get();
        }
    }
    // If not found, return the first device
    return m_deviceContexts.empty() ? nullptr : m_deviceContexts[0].get();
}

std::vector<int> ResourceManager::getAvailableDevices() const
{
    std::vector<int> devices;
    for (const auto& context : m_deviceContexts) {
        devices.push_back(context->getDeviceId());
    }
    return devices;
}

void ResourceManager::registerMaterialSystem(std::shared_ptr<IMaterialSystem> materialSystem)
{
    m_materialSystem = materialSystem;
}

// Legacy shader support functions have been removed
// These will be implemented differently for the resource system

ResourceID ResourceManager::generateResourceId()
{
    return m_nextResourceId++;
}

} // namespace optix_renderer