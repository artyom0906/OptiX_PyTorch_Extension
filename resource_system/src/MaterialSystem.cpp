#include "../include/MaterialSystem.h"
#include <iostream>
#include <cuda_runtime.h>

namespace optix_renderer {

bool MaterialSystem::initialize() {
    // Basic initialization of the material system
    std::cout << "Initializing MaterialSystem..." << std::endl;
    
    // Create standard material descriptions
    MaterialDescription lambertian;
    lambertian.name = "Standard Lambertian";
    lambertian.type = OPTIX_LAMBERTIAN;
    initializeMaterialDefaults(lambertian.defaultValues, OPTIX_LAMBERTIAN);
    materialDescriptions[nextHandle++] = lambertian;
    
    MaterialDescription pbr;
    pbr.name = "Standard PBR";
    pbr.type = OPTIX_PBR;
    initializeMaterialDefaults(pbr.defaultValues, OPTIX_PBR);
    materialDescriptions[nextHandle++] = pbr;
    
    MaterialDescription glass;
    glass.name = "Standard Glass";
    glass.type = OPTIX_GLASS;
    initializeMaterialDefaults(glass.defaultValues, OPTIX_GLASS);
    materialDescriptions[nextHandle++] = glass;
    
    MaterialDescription emissive;
    emissive.name = "Standard Emissive";
    emissive.type = OPTIX_EMISSIVE;
    initializeMaterialDefaults(emissive.defaultValues, OPTIX_EMISSIVE);
    materialDescriptions[nextHandle++] = emissive;
    
    MaterialDescription mirror;
    mirror.name = "Standard Mirror";
    mirror.type = OPTIX_MIRROR;
    initializeMaterialDefaults(mirror.defaultValues, OPTIX_MIRROR);
    materialDescriptions[nextHandle++] = mirror;
    
    std::cout << "MaterialSystem initialized with " << materialDescriptions.size() 
              << " standard materials" << std::endl;
    
    return true;
}

MaterialHandle MaterialSystem::loadMaterialFromFile(const std::string& filename, 
                                                  const std::string& materialName) {
    // Basic implementation without actual MDL loading for Phase 1
    std::cout << "Loading material from file: " << filename
              << ", material: " << materialName << std::endl;
    
    // Create a placeholder material description
    MaterialDescription desc;
    desc.name = materialName;
    desc.sourcePath = filename;
    desc.type = OPTIX_PBR; // Default to PBR for file-based materials
    
    // Initialize with default PBR values
    initializeMaterialDefaults(desc.defaultValues, OPTIX_PBR);
    
    // Store and return handle
    MaterialHandle handle = nextHandle++;
    materialDescriptions[handle] = desc;
    
    std::cout << "Created material description with handle " << handle << std::endl;
    return handle;
}

MaterialHandle MaterialSystem::createMaterialInstance(MaterialHandle materialDesc) {
    // Validate material description handle
    if (materialDescriptions.find(materialDesc) == materialDescriptions.end()) {
        std::cerr << "Invalid material description handle: " << materialDesc << std::endl;
        return 0;
    }
    
    // Create a new material instance
    MaterialInstance instance;
    instance.descriptionHandle = materialDesc;
    
    // Copy default values from description
    const auto& desc = materialDescriptions[materialDesc];
    instance.parameters = desc.defaultValues;
    
    // Store and return handle
    MaterialHandle handle = nextHandle++;
    materialInstances[handle] = instance;
    
    std::cout << "Created material instance with handle " << handle 
              << " from description " << materialDesc << std::endl;
    return handle;
}

bool MaterialSystem::setParameter(MaterialHandle material, 
                                const std::string& paramName, 
                                std::any value) {
    // Validate material handle
    if (materialInstances.find(material) == materialInstances.end()) {
        std::cerr << "Invalid material handle: " << material << std::endl;
        return false;
    }
    
    // Get the material instance
    auto& instance = materialInstances[material];
    
    // Convert the std::any value to the appropriate parameter
    try {
        convertAnyToMaterialParameter(value, paramName, instance.parameters);
        
        // Mark all device data as needing update
        for (auto& [device, data] : instance.deviceData) {
            data.needsUpdate = true;
        }
        
        // Update flags based on new parameters
        updateMaterialFlags(instance.parameters);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error setting parameter '" << paramName << "': " << e.what() << std::endl;
        return false;
    }
}

bool MaterialSystem::setTextureParameter(MaterialHandle material,
                                       const std::string& paramName,
                                       TextureHandle textureHandle) {
    // Validate material handle
    if (materialInstances.find(material) == materialInstances.end()) {
        std::cerr << "Invalid material handle: " << material << std::endl;
        return false;
    }
    
    // Get the material instance
    auto& instance = materialInstances[material];
    
    // Store the texture handle
    instance.textures[paramName] = textureHandle;
    
    // Set the appropriate texture index in the material parameters
    if (paramName == "albedo" || paramName == "baseColor") {
        instance.parameters.albedoTexture = static_cast<int>(textureHandle);
        setMaterialFlag(instance.parameters.flags, MATERIAL_HAS_ALBEDO_TEXTURE, true);
    } else if (paramName == "normal" || paramName == "normalMap") {
        instance.parameters.normalTexture = static_cast<int>(textureHandle);
        setMaterialFlag(instance.parameters.flags, MATERIAL_HAS_NORMAL_TEXTURE, true);
    } else if (paramName == "metallicRoughness") {
        instance.parameters.metallicRoughnessTexture = static_cast<int>(textureHandle);
        setMaterialFlag(instance.parameters.flags, MATERIAL_HAS_METALLIC_ROUGHNESS_TEXTURE, true);
    } else if (paramName == "emission" || paramName == "emissive") {
        instance.parameters.emissionTexture = static_cast<int>(textureHandle);
        setMaterialFlag(instance.parameters.flags, MATERIAL_HAS_EMISSION_TEXTURE, true);
    } else if (paramName == "specular") {
        instance.parameters.specularTexture = static_cast<int>(textureHandle);
    } else if (paramName == "specularTint") {
        instance.parameters.specularTintTexture = static_cast<int>(textureHandle);
    } else {
        std::cerr << "Unknown texture parameter: " << paramName << std::endl;
        return false;
    }
    
    // Mark all device data as needing update
    for (auto& [device, data] : instance.deviceData) {
        data.needsUpdate = true;
    }
    
    return true;
}

bool MaterialSystem::compileMaterial(MaterialHandle material,
                                   OptixDeviceContext deviceContext,
                                   void* compileOptions) {
    // In Phase 1, we don't need to do any actual compilation
    // We just need to allocate GPU memory for the material parameters
    
    // Validate material handle
    if (materialInstances.find(material) == materialInstances.end()) {
        std::cerr << "Invalid material handle: " << material << std::endl;
        return false;
    }
    
    // Get the material instance
    auto& instance = materialInstances[material];
    auto& deviceData = instance.deviceData[deviceContext];
    
    // Check if we need to update
    if (!deviceData.needsUpdate && deviceData.parameterBuffer != 0) {
        return true;  // Already up-to-date
    }
    
    // Free previous buffer if it exists
    if (deviceData.parameterBuffer != 0) {
        cuMemFree(deviceData.parameterBuffer);
        deviceData.parameterBuffer = 0;
    }
    
    // Allocate device memory for the material parameters
    const size_t paramSize = sizeof(SimpleMaterial);
    deviceData.parameterBufferSize = paramSize;
    
    CUresult result = cuMemAlloc(&deviceData.parameterBuffer, paramSize);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate device memory for material parameters" << std::endl;
        return false;
    }
    
    // Copy the material parameters to device memory
    result = cuMemcpyHtoD(deviceData.parameterBuffer, &instance.parameters, paramSize);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to copy material parameters to device memory" << std::endl;
        cuMemFree(deviceData.parameterBuffer);
        deviceData.parameterBuffer = 0;
        return false;
    }
    
    // Mark as up-to-date
    deviceData.needsUpdate = false;
    
    return true;
}

OptixProgramGroup MaterialSystem::getProgramGroup(MaterialHandle material,
                                               OptixDeviceContext deviceContext) {
    // In Phase 1, we don't use custom program groups for materials
    // We return nullptr to indicate that the default program group should be used
    return nullptr;
}

void* MaterialSystem::getParameterData(MaterialHandle material,
                                    OptixDeviceContext deviceContext) {
    // Validate material handle
    if (materialInstances.find(material) == materialInstances.end()) {
        std::cerr << "Invalid material handle: " << material << std::endl;
        return nullptr;
    }
    
    // Get the material instance
    auto& instance = materialInstances[material];
    
    // Check if we have device data for this context
    if (instance.deviceData.find(deviceContext) == instance.deviceData.end()) {
        // No device data yet, compile the material
        if (!compileMaterial(material, deviceContext, nullptr)) {
            return nullptr;
        }
    }
    
    // Return pointer to device memory - this will be a CUdeviceptr cast to void*
    return reinterpret_cast<void*>(instance.deviceData[deviceContext].parameterBuffer);
}

size_t MaterialSystem::getParameterDataSize(MaterialHandle material,
                                         OptixDeviceContext deviceContext) {
    // Validate material handle
    if (materialInstances.find(material) == materialInstances.end()) {
        std::cerr << "Invalid material handle: " << material << std::endl;
        return 0;
    }
    
    // Get the material instance
    auto& instance = materialInstances[material];
    
    // Check if we have device data for this context
    if (instance.deviceData.find(deviceContext) == instance.deviceData.end()) {
        // No device data yet, return the size of SimpleMaterial
        return sizeof(SimpleMaterial);
    }
    
    // Return the size of the device buffer
    return instance.deviceData[deviceContext].parameterBufferSize;
}

void MaterialSystem::initializeMaterialDefaults(SimpleMaterial& material, OptiXMaterialType type) {
    // Clear the material
    memset(&material, 0, sizeof(SimpleMaterial));
    
    // Set default values
    material.materialType = type;
    material.mdlMaterialId = -1;  // No MDL material by default
    material.flags = 0;
    
    // Common defaults
    material.albedo = make_float3(0.8f, 0.8f, 0.8f);
    material.emission = make_float3(0.0f, 0.0f, 0.0f);
    material.metallic = 0.0f;
    material.roughness = 0.5f;
    material.transmission = 0.0f;
    material.ior = 1.5f;
    material.specular = 0.5f;
    material.specularTint = 0.0f;
    material.sheen = 0.0f;
    material.sheenTint = 0.5f;
    material.clearcoat = 0.0f;
    material.clearcoatGloss = 0.0f;
    material.anisotropic = 0.0f;
    
    // Set all texture indices to -1 (not used)
    material.albedoTexture = -1;
    material.normalTexture = -1;
    material.metallicRoughnessTexture = -1;
    material.emissionTexture = -1;
    material.specularTexture = -1;
    material.specularTintTexture = -1;
    
    // Set type-specific defaults
    switch (type) {
        case OPTIX_LAMBERTIAN:
            // Lambertian defaults already set
            break;
            
        case OPTIX_PBR:
            // PBR defaults
            material.metallic = 0.0f;
            material.roughness = 0.5f;
            material.specular = 0.5f;
            break;
            
        case OPTIX_GLASS:
            // Glass defaults
            material.transmission = 1.0f;
            material.ior = 1.5f;
            material.roughness = 0.0f;
            setMaterialFlag(material.flags, MATERIAL_IS_TRANSPARENT, true);
            break;
            
        case OPTIX_EMISSIVE:
            // Emissive defaults
            material.emission = make_float3(1.0f, 0.9f, 0.8f);
            setMaterialFlag(material.flags, MATERIAL_IS_EMISSIVE, true);
            break;
            
        case OPTIX_MIRROR:
            // Mirror defaults
            material.metallic = 1.0f;
            material.roughness = 0.0f;
            material.specular = 1.0f;
            break;
            
        case OPTIX_MDL:
            // MDL defaults - will be overridden by actual MDL material
            setMaterialFlag(material.flags, MATERIAL_IS_MDL, true);
            break;
    }
}

void MaterialSystem::convertAnyToMaterialParameter(std::any value, const std::string& paramName, 
                                                 SimpleMaterial& material) {
    // Handle float parameters
    if (paramName == "metallic") {
        material.metallic = std::any_cast<float>(value);
    } else if (paramName == "roughness") {
        material.roughness = std::any_cast<float>(value);
    } else if (paramName == "transmission") {
        material.transmission = std::any_cast<float>(value);
        setMaterialFlag(material.flags, MATERIAL_IS_TRANSPARENT, material.transmission > 0.0f);
    } else if (paramName == "ior") {
        material.ior = std::any_cast<float>(value);
    } else if (paramName == "specular") {
        material.specular = std::any_cast<float>(value);
    } else if (paramName == "specularTint") {
        material.specularTint = std::any_cast<float>(value);
    } else if (paramName == "sheen") {
        material.sheen = std::any_cast<float>(value);
        setMaterialFlag(material.flags, MATERIAL_HAS_SHEEN, material.sheen > 0.0f);
    } else if (paramName == "sheenTint") {
        material.sheenTint = std::any_cast<float>(value);
    } else if (paramName == "clearcoat") {
        material.clearcoat = std::any_cast<float>(value);
        setMaterialFlag(material.flags, MATERIAL_HAS_CLEARCOAT, material.clearcoat > 0.0f);
    } else if (paramName == "clearcoatGloss") {
        material.clearcoatGloss = std::any_cast<float>(value);
    } else if (paramName == "anisotropic") {
        material.anisotropic = std::any_cast<float>(value);
        setMaterialFlag(material.flags, MATERIAL_HAS_ANISOTROPY, material.anisotropic > 0.0f);
    } 
    // Handle float3 parameters
    else if (paramName == "albedo" || paramName == "baseColor") {
        if (value.type() == typeid(float3)) {
            material.albedo = std::any_cast<float3>(value);
        } else if (value.type() == typeid(std::vector<float>)) {
            auto vec = std::any_cast<std::vector<float>>(value);
            if (vec.size() >= 3) {
                material.albedo = make_float3(vec[0], vec[1], vec[2]);
            }
        }
    } else if (paramName == "emission" || paramName == "emissive") {
        if (value.type() == typeid(float3)) {
            material.emission = std::any_cast<float3>(value);
        } else if (value.type() == typeid(std::vector<float>)) {
            auto vec = std::any_cast<std::vector<float>>(value);
            if (vec.size() >= 3) {
                material.emission = make_float3(vec[0], vec[1], vec[2]);
            }
        }
        // Check if emission is non-zero
        if (material.emission.x > 0.0f || material.emission.y > 0.0f || material.emission.z > 0.0f) {
            setMaterialFlag(material.flags, MATERIAL_IS_EMISSIVE, true);
        } else {
            setMaterialFlag(material.flags, MATERIAL_IS_EMISSIVE, false);
        }
    }
    // Handle integer parameters
    else if (paramName == "materialType") {
        material.materialType = std::any_cast<int>(value);
    } else if (paramName == "mdlMaterialId") {
        material.mdlMaterialId = std::any_cast<int>(value);
        setMaterialFlag(material.flags, MATERIAL_IS_MDL, material.mdlMaterialId >= 0);
    }
    // Handle texture indices
    else if (paramName == "albedoTexture") {
        material.albedoTexture = std::any_cast<int>(value);
        setMaterialFlag(material.flags, MATERIAL_HAS_ALBEDO_TEXTURE, material.albedoTexture >= 0);
    } else if (paramName == "normalTexture") {
        material.normalTexture = std::any_cast<int>(value);
        setMaterialFlag(material.flags, MATERIAL_HAS_NORMAL_TEXTURE, material.normalTexture >= 0);
    } else if (paramName == "metallicRoughnessTexture") {
        material.metallicRoughnessTexture = std::any_cast<int>(value);
        setMaterialFlag(material.flags, MATERIAL_HAS_METALLIC_ROUGHNESS_TEXTURE, 
                       material.metallicRoughnessTexture >= 0);
    } else if (paramName == "emissionTexture") {
        material.emissionTexture = std::any_cast<int>(value);
        setMaterialFlag(material.flags, MATERIAL_HAS_EMISSION_TEXTURE, material.emissionTexture >= 0);
    } else if (paramName == "specularTexture") {
        material.specularTexture = std::any_cast<int>(value);
    } else if (paramName == "specularTintTexture") {
        material.specularTintTexture = std::any_cast<int>(value);
    } else {
        throw std::runtime_error("Unknown parameter name: " + paramName);
    }
}

bool MaterialSystem::updateMaterialFlags(SimpleMaterial& material) {
    // Update transparent flag
    setMaterialFlag(material.flags, MATERIAL_IS_TRANSPARENT, material.transmission > 0.0f);
    
    // Update emissive flag
    bool isEmissive = material.emission.x > 0.0f || material.emission.y > 0.0f || material.emission.z > 0.0f;
    setMaterialFlag(material.flags, MATERIAL_IS_EMISSIVE, isEmissive);
    
    // Update texture flags
    setMaterialFlag(material.flags, MATERIAL_HAS_ALBEDO_TEXTURE, material.albedoTexture >= 0);
    setMaterialFlag(material.flags, MATERIAL_HAS_NORMAL_TEXTURE, material.normalTexture >= 0);
    setMaterialFlag(material.flags, MATERIAL_HAS_METALLIC_ROUGHNESS_TEXTURE, material.metallicRoughnessTexture >= 0);
    setMaterialFlag(material.flags, MATERIAL_HAS_EMISSION_TEXTURE, material.emissionTexture >= 0);
    
    // Update effect flags
    setMaterialFlag(material.flags, MATERIAL_HAS_SHEEN, material.sheen > 0.0f);
    setMaterialFlag(material.flags, MATERIAL_HAS_CLEARCOAT, material.clearcoat > 0.0f);
    setMaterialFlag(material.flags, MATERIAL_HAS_ANISOTROPY, material.anisotropic > 0.0f);
    
    // Update MDL flag
    setMaterialFlag(material.flags, MATERIAL_IS_MDL, material.mdlMaterialId >= 0);
    
    return true;
}

// IMaterialSystem interface implementation
MaterialHandle MaterialSystem::loadMaterial(const std::string& path, const std::string& name) {
    // Forward to the more specific implementation
    return loadMaterialFromFile(path, name);
}

bool MaterialSystem::compileMaterial(MaterialHandle handle, int deviceId) {
    // We need a device context to call the more specific implementation
    // In our case, we don't have direct access to device contexts, so we'll return false
    // The caller should use the other compileMaterial method with an OptixDeviceContext parameter
    std::cerr << "Warning: compileMaterial(handle, deviceId) is not directly supported." << std::endl;
    std::cerr << "Use compileMaterial(handle, deviceContext, options) instead." << std::endl;
    return false;
}

std::vector<MaterialParameterInfo> MaterialSystem::getMaterialParameters(MaterialHandle handle) {
    std::vector<MaterialParameterInfo> parameters;
    
    // Validate material handle
    if (materialInstances.find(handle) == materialInstances.end() && 
        materialDescriptions.find(handle) == materialDescriptions.end()) {
        std::cerr << "Invalid material handle: " << handle << std::endl;
        return parameters;
    }
    
    // Define standard parameters based on material type
    // For simplicity, we'll define a fixed set of common parameters
    MaterialParameterInfo albedo;
    albedo.name = "albedo";
    albedo.displayName = "Base Color";
    albedo.type = ParameterType::FLOAT3;
    albedo.defaultValue = MaterialParameter(make_float3(0.8f, 0.8f, 0.8f));
    parameters.push_back(albedo);
    
    MaterialParameterInfo metallic;
    metallic.name = "metallic";
    metallic.displayName = "Metallic";
    metallic.type = ParameterType::FLOAT;
    metallic.defaultValue = MaterialParameter(0.0f);
    metallic.minValue = 0.0f;
    metallic.maxValue = 1.0f;
    parameters.push_back(metallic);
    
    MaterialParameterInfo roughness;
    roughness.name = "roughness";
    roughness.displayName = "Roughness";
    roughness.type = ParameterType::FLOAT;
    roughness.defaultValue = MaterialParameter(0.5f);
    roughness.minValue = 0.0f;
    roughness.maxValue = 1.0f;
    parameters.push_back(roughness);
    
    MaterialParameterInfo emission;
    emission.name = "emission";
    emission.displayName = "Emission";
    emission.type = ParameterType::FLOAT3;
    emission.defaultValue = MaterialParameter(make_float3(0.0f, 0.0f, 0.0f));
    parameters.push_back(emission);
    
    // Add more parameters depending on material type
    OptiXMaterialType matType = OPTIX_PBR;  // Default
    
    // Determine material type
    if (materialInstances.find(handle) != materialInstances.end()) {
        const auto& desc = materialDescriptions[materialInstances[handle].descriptionHandle];
        matType = desc.type;
    } else if (materialDescriptions.find(handle) != materialDescriptions.end()) {
        matType = materialDescriptions[handle].type;
    }
    
    // Add material-specific parameters
    if (matType == OPTIX_GLASS || matType == OPTIX_PBR) {
        MaterialParameterInfo ior;
        ior.name = "ior";
        ior.displayName = "Index of Refraction";
        ior.type = ParameterType::FLOAT;
        ior.defaultValue = MaterialParameter(1.5f);
        ior.minValue = 1.0f;
        ior.maxValue = 3.0f;
        parameters.push_back(ior);
    }
    
    if (matType == OPTIX_GLASS) {
        MaterialParameterInfo transmission;
        transmission.name = "transmission";
        transmission.displayName = "Transmission";
        transmission.type = ParameterType::FLOAT;
        transmission.defaultValue = MaterialParameter(1.0f);
        transmission.minValue = 0.0f;
        transmission.maxValue = 1.0f;
        parameters.push_back(transmission);
    }
    
    if (matType == OPTIX_PBR) {
        MaterialParameterInfo specular;
        specular.name = "specular";
        specular.displayName = "Specular";
        specular.type = ParameterType::FLOAT;
        specular.defaultValue = MaterialParameter(0.5f);
        specular.minValue = 0.0f;
        specular.maxValue = 1.0f;
        parameters.push_back(specular);
        
        MaterialParameterInfo anisotropic;
        anisotropic.name = "anisotropic";
        anisotropic.displayName = "Anisotropic";
        anisotropic.type = ParameterType::FLOAT;
        anisotropic.defaultValue = MaterialParameter(0.0f);
        anisotropic.minValue = 0.0f;
        anisotropic.maxValue = 1.0f;
        parameters.push_back(anisotropic);
    }
    
    return parameters;
}

void MaterialSystem::setMaterialParameter(MaterialHandle handle, const std::string& name, const MaterialParameter& value) {
    // Validate material handle
    if (materialInstances.find(handle) == materialInstances.end()) {
        std::cerr << "Invalid material handle: " << handle << std::endl;
        return;
    }
    
    // Get the material instance
    auto& instance = materialInstances[handle];
    
    // Handle different parameter types
    switch (value.getType()) {
        case ParameterType::FLOAT:
            setParameter(handle, name, value.asFloat());
            break;
            
        case ParameterType::FLOAT2:
            {
                float2 val = value.asFloat2();
                std::vector<float> vec = {val.x, val.y};
                setParameter(handle, name, vec);
            }
            break;
            
        case ParameterType::FLOAT3:
            {
                float3 val = value.asFloat3();
                std::vector<float> vec = {val.x, val.y, val.z};
                setParameter(handle, name, vec);
            }
            break;
            
        case ParameterType::FLOAT4:
            {
                float4 val = value.asFloat4();
                std::vector<float> vec = {val.x, val.y, val.z, val.w};
                setParameter(handle, name, vec);
            }
            break;
            
        case ParameterType::INT:
            setParameter(handle, name, value.asInt());
            break;
            
        case ParameterType::BOOL:
            setParameter(handle, name, value.asBool() ? 1.0f : 0.0f);
            break;
            
        case ParameterType::TEXTURE:
            setTextureParameter(handle, name, value.asTexture());
            break;
            
        case ParameterType::COLOR:
            {
                float3 val = value.asFloat3();
                std::vector<float> vec = {val.x, val.y, val.z};
                setParameter(handle, name, vec);
            }
            break;
    }
}

} // namespace optix_renderer