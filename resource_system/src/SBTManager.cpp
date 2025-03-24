#include "../include/SBTManager.h"

namespace optix_renderer {

// Constructor
SBTManager::SBTManager(ResourceManager* resourceManager, DeviceContext* deviceContext)
    : m_resourceManager(resourceManager),
      m_deviceContext(deviceContext),
      m_d_raygen_record(0),
      m_d_miss_record(0),
      m_d_hitgroup_records(0),
      m_hitgroup_record_size(0),
      m_hitgroup_record_count(0),
      m_prev_hitgroup_record_count(0)
{
    // Initialize SBT with zeros
    memset(&m_sbt, 0, sizeof(OptixShaderBindingTable));
}

// Destructor
SBTManager::~SBTManager() {
    cleanup();
}

// Cleanup resources
void SBTManager::cleanup() {
    // First ensure we're using the correct CUDA context
    if (!m_deviceContext->setDevice()) {
        std::cerr << "Failed to set device context for SBT cleanup" << std::endl;
        return;
    }

    // Clean up ray generation record
    if (m_d_raygen_record) {
        CUDA_DRIVER_CHECK(cuMemFreeAsync(m_d_raygen_record, m_deviceContext->getStream()));
        m_d_raygen_record = 0;
    }

    // Clean up miss record
    if (m_d_miss_record) {
        CUDA_DRIVER_CHECK(cuMemFreeAsync(m_d_miss_record, m_deviceContext->getStream()));
        m_d_miss_record = 0;
    }

    // Clean up hit group records
    cleanupHitgroupRecords();

    // Reset SBT
    memset(&m_sbt, 0, sizeof(OptixShaderBindingTable));
}

// Clean up hit group records
void SBTManager::cleanupHitgroupRecords() {
    if (m_d_hitgroup_records) {
        CUDA_DRIVER_CHECK(cuMemFreeAsync(m_d_hitgroup_records, m_deviceContext->getStream()));
        m_d_hitgroup_records = 0;
    }
    m_prev_hitgroup_record_count = m_hitgroup_record_count;
    m_hitgroup_record_count = 0;
}

// Allocate hit group records
bool SBTManager::allocateHitgroupRecords(size_t count) {
    if (count == 0) {
        return true; // Nothing to allocate
    }

    // Clean up previous records
    cleanupHitgroupRecords();

    // Calculate aligned size
    m_hitgroup_record_size = sizeof(HitGroupSbtRecord);
    size_t hitgroup_record_aligned_size = OPTIX_SBT_RECORD_ALIGNMENT * 
        ((m_hitgroup_record_size + OPTIX_SBT_RECORD_ALIGNMENT - 1) / OPTIX_SBT_RECORD_ALIGNMENT);

    // Allocate new records
    try {
        CUDA_DRIVER_CHECK(cuMemAllocAsync(&m_d_hitgroup_records, 
            count * hitgroup_record_aligned_size, 
            m_deviceContext->getStream()));
        
        m_hitgroup_record_count = count;
        
        // Verify alignment
        if ((m_d_hitgroup_records % OPTIX_SBT_RECORD_ALIGNMENT) != 0) {
            std::cerr << "ERROR: Hit group records not aligned to OPTIX_SBT_RECORD_ALIGNMENT" << std::endl;
            cleanupHitgroupRecords();
            return false;
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to allocate hit group records: " << e.what() << std::endl;
        cleanupHitgroupRecords();
        return false;
    }
}

// Build or rebuild the SBT
bool SBTManager::buildSBT(
    const std::vector<std::shared_ptr<GeometryInstance>>& instances,
    OptixProgramGroup raygenPG,
    OptixProgramGroup missPG,
    OptixProgramGroup hitgroupPG
) {
    // First ensure we're using the correct CUDA context
    if (!m_deviceContext->setDevice()) {
        std::cerr << "Failed to set device context for SBT setup" << std::endl;
        return false;
    }

    try {
        // Clean up existing SBT
        cleanup();
        
        // Calculate record sizes and alignment
        size_t raygen_record_size = sizeof(RayGenSbtRecord);
        size_t miss_record_size = sizeof(MissSbtRecord);
        
        size_t raygen_record_aligned_size = OPTIX_SBT_RECORD_ALIGNMENT * 
            ((raygen_record_size + OPTIX_SBT_RECORD_ALIGNMENT - 1) / OPTIX_SBT_RECORD_ALIGNMENT);
        size_t miss_record_aligned_size = OPTIX_SBT_RECORD_ALIGNMENT * 
            ((miss_record_size + OPTIX_SBT_RECORD_ALIGNMENT - 1) / OPTIX_SBT_RECORD_ALIGNMENT);
        
        // Allocate device memory for ray gen and miss records
        CUDA_DRIVER_CHECK(cuMemAllocAsync(&m_d_raygen_record, raygen_record_aligned_size, m_deviceContext->getStream()));
        CUDA_DRIVER_CHECK(cuMemAllocAsync(&m_d_miss_record, miss_record_aligned_size, m_deviceContext->getStream()));
        
        // Verify alignment
        if ((m_d_raygen_record % OPTIX_SBT_RECORD_ALIGNMENT) != 0 ||
            (m_d_miss_record % OPTIX_SBT_RECORD_ALIGNMENT) != 0) {
            std::cerr << "ERROR: SBT records not aligned to OPTIX_SBT_RECORD_ALIGNMENT" << std::endl;
            cleanup();
            return false;
        }
        
        // Create and fill raygen and miss records on host
        RayGenSbtRecord raygen_data;
        MissSbtRecord miss_data;
        
        // Pack headers
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &raygen_data.header));
        OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &miss_data.header));
        
        // Copy to device
        CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(m_d_raygen_record, &raygen_data, raygen_record_size, m_deviceContext->getStream()));
        CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(m_d_miss_record, &miss_data, miss_record_size, m_deviceContext->getStream()));
        
        // Allocate and set up hit group records
        if (!allocateHitgroupRecords(instances.size())) {
            std::cerr << "ERROR: Failed to allocate hit group records" << std::endl;
            cleanup();
            return false;
        }
        
        // Set up SBT entries for hit groups
        if (instances.size() > 0) {
            // Create host-side buffer for all hit group records
            std::vector<HitGroupSbtRecord> hitgroup_records(instances.size());
            
            // Initialize all hit group records with default values
            for (size_t i = 0; i < instances.size(); i++) {
                // Pack hit group header
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, &hitgroup_records[i].header));
                
                // Populate the record with instance data
                if (!populateHitgroupRecord(hitgroup_records[i], instances[i])) {
                    std::cerr << "ERROR: Failed to populate hit group record for instance " << i << std::endl;
                    cleanup();
                    return false;
                }
            }
            
            // Copy all hit group records to device
            size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
            CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(
                m_d_hitgroup_records,
                hitgroup_records.data(),
                hitgroup_records.size() * hitgroup_record_size,
                m_deviceContext->getStream()
            ));
        }
        
        // Setup final SBT structure
        m_sbt.raygenRecord = m_d_raygen_record;
        m_sbt.missRecordBase = m_d_miss_record;
        m_sbt.missRecordCount = 1;
        m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        
        m_sbt.hitgroupRecordBase = m_d_hitgroup_records;
        m_sbt.hitgroupRecordCount = static_cast<unsigned int>(instances.size());
        m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: SBT setup failed: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

// Update instance data without full rebuild
bool SBTManager::updateInstanceData(const std::vector<std::shared_ptr<GeometryInstance>>& instances) {
    // First ensure we're using the correct CUDA context
    if (!m_deviceContext->setDevice()) {
        std::cerr << "Failed to set device context for SBT update" << std::endl;
        return false;
    }
    
    // If count changed, we need a full rebuild
    if (instances.size() != m_hitgroup_record_count) {
        std::cout << "Instance count changed, performing full SBT rebuild" << std::endl;
        return false; // Caller should do full rebuild
    }
    
    try {
        // Update all hit group records
        if (instances.size() > 0) {
            // Create host-side buffer for all hit group records
            std::vector<HitGroupSbtRecord> hitgroup_records(instances.size());
            
            // First copy existing SBT records from device to host
            size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
            CUDA_DRIVER_CHECK(cuMemcpyDtoHAsync(
                hitgroup_records.data(),
                m_d_hitgroup_records,
                hitgroup_records.size() * hitgroup_record_size,
                m_deviceContext->getStream()
            ));
            
            // Wait for copy to complete
            CUDA_DRIVER_CHECK(cuStreamSynchronize(m_deviceContext->getStream()));
            
            // Update only instance-specific data, keep headers intact
            for (size_t i = 0; i < instances.size(); i++) {
                // Populate the record with updated instance data
                if (!populateHitgroupRecord(hitgroup_records[i], instances[i])) {
                    std::cerr << "ERROR: Failed to update hit group record for instance " << i << std::endl;
                    return false;
                }
            }
            
            // Copy all updated hit group records back to device
            CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(
                m_d_hitgroup_records,
                hitgroup_records.data(),
                hitgroup_records.size() * hitgroup_record_size,
                m_deviceContext->getStream()
            ));
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: SBT update failed: " << e.what() << std::endl;
        return false;
    }
}

// Populate a hit group record with instance data
bool SBTManager::populateHitgroupRecord(HitGroupSbtRecord& record, std::shared_ptr<GeometryInstance> instance) {
    if (!instance) {
        std::cerr << "ERROR: NULL instance pointer" << std::endl;
        return false;
    }
    
    // Get geometry and material handles
    GeometryHandle geomId = instance->getGeometryHandle();
    MaterialHandle matId = instance->getMaterialHandle();
    
    // Initialize with default values
    record.data.material.albedo = make_float3(0.8f, 0.8f, 0.8f);
    record.data.material.emission = make_float3(0.0f, 0.0f, 0.0f);
    record.data.material.metallic = 0.0f;
    record.data.material.roughness = 0.5f;
    record.data.material.transmission = 0.0f;
    record.data.material.ior = 1.5f;
    record.data.material.materialType = OPTIX_LAMBERTIAN;
    record.data.material.flags = 0;
    
    record.data.vertices = 0;
    record.data.indices = 0;
    record.data.normals = 0;
    record.data.texcoords = 0;
    record.data.tangents = 0;
    record.data.bitangents = 0;
    
    record.data.albedo_texture = 0;
    record.data.normal_texture = 0;
    record.data.metallic_roughness_texture = 0;
    record.data.emission_texture = 0;
    record.data.specular_texture = 0;
    record.data.specular_tint_texture = 0;
    record.data.sheen_texture = 0;
    record.data.clearcoat_texture = 0;
    
    record.data.has_normals = false;
    record.data.has_texcoords = false;
    record.data.has_tangents = false;
    record.data.has_bitangents = false;
    
    record.data.albedo = make_float3(0.8f, 0.8f, 0.8f);
    record.data.emission = make_float3(0.0f, 0.0f, 0.0f);
    
    // Update geometry data
    auto geomData = m_deviceContext->getGeometryData(geomId);
    if (geomData) {
        // Set geometry device pointers
        record.data.vertices = geomData->d_vertices;
        record.data.indices = geomData->d_indices;
        record.data.texcoords = geomData->d_texCoords;
        
        // Set geometry flags
        record.data.has_texcoords = (geomData->d_texCoords != 0);
    }
    
    // Update material data
    auto matResource = m_resourceManager->getMaterial(matId);
    if (matResource) {
        // Set material type
        MaterialType materialType = matResource->getMaterialType();
        
        // Map material type to OptiX shader's material type
        int optixMaterialType;
        switch(materialType) {
            case MaterialType::LAMBERTIAN: optixMaterialType = OPTIX_LAMBERTIAN; break;
            case MaterialType::PBR: optixMaterialType = OPTIX_PBR; break;
            case MaterialType::GLASS: optixMaterialType = OPTIX_GLASS; break;
            case MaterialType::EMISSIVE: optixMaterialType = OPTIX_EMISSIVE; break;
            case MaterialType::MIRROR: optixMaterialType = OPTIX_MIRROR; break;
            default: optixMaterialType = OPTIX_LAMBERTIAN; break;
        }
        
        record.data.material.materialType = optixMaterialType;
        
        // Set base material parameters
        // Look for albedo/base color with multiple possible names
        const std::vector<std::string> albedoParamNames = {
            "albedo", "base_color", "diffuse_color", "color"
        };
        
        for (const auto& paramName : albedoParamNames) {
            if (matResource->hasParameter(paramName)) {
                auto param = matResource->getParameter(paramName);
                if (param.getType() == ParameterType::FLOAT3) {
                    float3 value = param.asFloat3();
                    record.data.material.albedo = value;
                    record.data.albedo = value; // Set both for compatibility
                    break;
                }
            }
        }
        
        // Handle emission parameters
        if (matResource->hasParameter("emission") || matResource->hasParameter("emission_color")) {
            const std::string paramName = matResource->hasParameter("emission") ? 
                                        "emission" : "emission_color";
            auto param = matResource->getParameter(paramName);
            if (param.getType() == ParameterType::FLOAT3) {
                float3 value = param.asFloat3();
                record.data.material.emission = value;
                record.data.emission = value; // Set both for compatibility
                
                // If emission is non-zero, mark material as emissive
                if (value.x > 0.0f || value.y > 0.0f || value.z > 0.0f) {
                    setMaterialFlag(record.data.material.flags, MATERIAL_IS_EMISSIVE, true);
                }
            }
        }
        
        // Handle emission strength
        if (matResource->hasParameter("emission_strength")) {
            auto param = matResource->getParameter("emission_strength");
            if (param.getType() == ParameterType::FLOAT) {
                float strength = param.asFloat();
                // Scale emission color by strength
                record.data.material.emission.x *= strength;
                record.data.material.emission.y *= strength;
                record.data.material.emission.z *= strength;
                record.data.emission = record.data.material.emission;
                
                // If strength is non-zero, mark material as emissive
                if (strength > 0.0f) {
                    setMaterialFlag(record.data.material.flags, MATERIAL_IS_EMISSIVE, true);
                }
            }
        }
        
        // Handle PBR parameters
        // Metallic
        if (matResource->hasParameter("metallic") || matResource->hasParameter("metalness")) {
            const std::string paramName = matResource->hasParameter("metallic") ? 
                                        "metallic" : "metalness";
            auto param = matResource->getParameter(paramName);
            if (param.getType() == ParameterType::FLOAT) {
                record.data.material.metallic = param.asFloat();
            }
        }
        
        // Roughness
        if (matResource->hasParameter("roughness")) {
            auto param = matResource->getParameter("roughness");
            if (param.getType() == ParameterType::FLOAT) {
                record.data.material.roughness = param.asFloat();
            }
        }
        
        // Handle textures
        configureTexturesForMaterial(record, matResource, 0);
    }
    
    return true;
}

// Handle all texture configuration for materials
void SBTManager::configureTexturesForMaterial(HitGroupSbtRecord& record, MaterialResource* matResource, size_t instanceIdx) {
    if (!matResource) {
        return;
    }
    
    // Initialize all texture indices to -1 (no texture)
    record.data.material.albedoTexture = -1;
    record.data.material.normalTexture = -1;
    record.data.material.metallicRoughnessTexture = -1;
    record.data.material.emissionTexture = -1;
    record.data.material.specularTexture = -1;
    record.data.material.specularTintTexture = -1;
    
    // Albedo/Base Color texture (multiple possible names)
    const std::vector<std::string> albedoTextureNames = {
        "albedoTexture", "albedo_texture", "baseColorTexture", "diffuseTexture", "colorTexture"
    };
    
    for (const auto& texName : albedoTextureNames) {
        if (matResource->hasTexture(texName)) {
            TextureHandle texHandle = matResource->getTexture(texName);
            
            // Get the texture data from the device context
            auto texData = m_deviceContext->getTextureData(texHandle);
            if (texData && texData->texture_object) {
                // Update texture in SBT
                record.data.albedo_texture = texData->texture_object;
                record.data.material.albedoTexture = 0;  // Set to 0 to indicate texture is used (not -1)
                
                // Set material flag to indicate texture is available
                setMaterialFlag(record.data.material.flags, MATERIAL_HAS_ALBEDO_TEXTURE, true);
                
                // Found a valid texture, stop checking other names
                break;
            }
        }
    }
    
    // Normal map texture
    if (matResource->hasTexture("normalTexture") || matResource->hasTexture("normal_texture")) {
        TextureHandle texHandle = matResource->hasTexture("normalTexture") ? 
            matResource->getTexture("normalTexture") : matResource->getTexture("normal_texture");
        
        auto texData = m_deviceContext->getTextureData(texHandle);
        if (texData && texData->texture_object) {
            record.data.normal_texture = texData->texture_object;
            record.data.material.normalTexture = 0;
            setMaterialFlag(record.data.material.flags, MATERIAL_HAS_NORMAL_TEXTURE, true);
        }
    }
    
    // Metallic-roughness texture
    if (matResource->hasTexture("metallicRoughnessTexture") || 
        matResource->hasTexture("metallic_roughness_texture")) {
        
        TextureHandle texHandle = matResource->hasTexture("metallicRoughnessTexture") ? 
            matResource->getTexture("metallicRoughnessTexture") : 
            matResource->getTexture("metallic_roughness_texture");
        
        auto texData = m_deviceContext->getTextureData(texHandle);
        if (texData && texData->texture_object) {
            record.data.metallic_roughness_texture = texData->texture_object;
            record.data.material.metallicRoughnessTexture = 0;
            setMaterialFlag(record.data.material.flags, MATERIAL_HAS_METALLIC_ROUGHNESS_TEXTURE, true);
        }
    }
    
    // Emission texture
    if (matResource->hasTexture("emissionTexture") || matResource->hasTexture("emission_texture")) {
        TextureHandle texHandle = matResource->hasTexture("emissionTexture") ? 
            matResource->getTexture("emissionTexture") : matResource->getTexture("emission_texture");
        
        auto texData = m_deviceContext->getTextureData(texHandle);
        if (texData && texData->texture_object) {
            record.data.emission_texture = texData->texture_object;
            record.data.material.emissionTexture = 0;
            setMaterialFlag(record.data.material.flags, MATERIAL_HAS_EMISSION_TEXTURE, true);
            
            // Also mark the material as emissive
            setMaterialFlag(record.data.material.flags, MATERIAL_IS_EMISSIVE, true);
        }
    }
    
    // Additional PBR textures
    // Specular texture
    if (matResource->hasTexture("specularTexture") || matResource->hasTexture("specular_texture")) {
        TextureHandle texHandle = matResource->hasTexture("specularTexture") ? 
            matResource->getTexture("specularTexture") : matResource->getTexture("specular_texture");
        
        auto texData = m_deviceContext->getTextureData(texHandle);
        if (texData && texData->texture_object) {
            record.data.specular_texture = texData->texture_object;
            record.data.material.specularTexture = 0;
        }
    }
}

} // namespace optix_renderer