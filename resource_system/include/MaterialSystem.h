#pragma once

#include <unordered_map>
#include <string>
#include <any>
#include <memory>
#include <vector>
#include <cuda.h>
#include <optix.h>

#include "IMaterialSystem.h"
#include "ResourceTypes.h"
#include "TextureResource.h"
#include "../src/kernels/material_types.h"

namespace optix_renderer {

/**
 * Implementation of the material system that manages PBR materials
 * with support for textures and MDL compatibility
 */
class MaterialSystem : public IMDLMaterialSystem {
public:
    MaterialSystem() = default;
    ~MaterialSystem() override = default;
    
    // IMaterialSystem implementation
    virtual MaterialHandle loadMaterial(const std::string& path, const std::string& name) override;
    virtual bool compileMaterial(MaterialHandle handle, int deviceId) override;
    virtual std::vector<MaterialParameterInfo> getMaterialParameters(MaterialHandle handle) override;
    virtual void setMaterialParameter(MaterialHandle handle, const std::string& name, const MaterialParameter& value) override;

    // IMDLMaterialSystem implementation
    bool initialize() override;
    MaterialHandle loadMaterialFromFile(const std::string& filename, 
                                      const std::string& materialName) override;
    MaterialHandle createMaterialInstance(MaterialHandle materialDesc) override;
    bool setParameter(MaterialHandle material, 
                    const std::string& paramName, 
                    std::any value) override;
    bool setTextureParameter(MaterialHandle material,
                           const std::string& paramName,
                           TextureHandle textureHandle) override;
    bool compileMaterial(MaterialHandle material,
                       OptixDeviceContext deviceContext,
                       void* compileOptions) override;
    OptixProgramGroup getProgramGroup(MaterialHandle material,
                                   OptixDeviceContext deviceContext) override;
    void* getParameterData(MaterialHandle material,
                        OptixDeviceContext deviceContext) override;
    size_t getParameterDataSize(MaterialHandle material,
                             OptixDeviceContext deviceContext) override;
    
    // Additional utility methods
    void initializeMaterialDefaults(SimpleMaterial& material, OptiXMaterialType type);
    
private:
    struct MaterialDescription {
        std::string name;
        std::string sourcePath;
        OptiXMaterialType type;
        SimpleMaterial defaultValues;
    };
    
    struct MaterialInstance {
        MaterialHandle descriptionHandle;
        SimpleMaterial parameters;
        std::unordered_map<std::string, TextureHandle> textures;
        
        // Device-specific compiled data
        struct DeviceData {
            CUdeviceptr parameterBuffer = 0;
            size_t parameterBufferSize = 0;
            OptixProgramGroup programGroup = nullptr;
            bool needsUpdate = true;
        };
        std::unordered_map<OptixDeviceContext, DeviceData> deviceData;
    };
    
    // Internal storage for materials
    std::unordered_map<MaterialHandle, MaterialDescription> materialDescriptions;
    std::unordered_map<MaterialHandle, MaterialInstance> materialInstances;
    MaterialHandle nextHandle = 1; // Start from 1 so that 0 can represent an invalid handle

    // Helper methods
    void convertAnyToMaterialParameter(std::any value, const std::string& paramName, 
                                     SimpleMaterial& material);
    bool updateMaterialFlags(SimpleMaterial& material);
};

} // namespace optix_renderer