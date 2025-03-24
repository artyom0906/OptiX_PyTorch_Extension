#pragma once

#include <optix.h>
#include <vector>
#include <memory>
#include "LaunchParams.h"
#include "ResourceManager.cuh"
#include "DeviceContext.cuh"
#include "GeometryInstance.h"

namespace optix_renderer {

// Manages creation and updates of the OptiX Shader Binding Table (SBT)
class SBTManager {
public:
    SBTManager(ResourceManager* resourceManager, DeviceContext* deviceContext);
    ~SBTManager();

    // Build or rebuild the SBT
    bool buildSBT(
        const std::vector<std::shared_ptr<GeometryInstance>>& instances,
        OptixProgramGroup raygenPG,
        OptixProgramGroup missPG,
        OptixProgramGroup hitgroupPG
    );

    // Update just instance data without full rebuild
    bool updateInstanceData(const std::vector<std::shared_ptr<GeometryInstance>>& instances);

    // Getter for the SBT
    const OptixShaderBindingTable& getSBT() const { return m_sbt; }

    // Clean up resources
    void cleanup();

private:
    ResourceManager* m_resourceManager;
    DeviceContext* m_deviceContext;
    OptixShaderBindingTable m_sbt;

    // Device memory for SBT records
    CUdeviceptr m_d_raygen_record;
    CUdeviceptr m_d_miss_record;
    CUdeviceptr m_d_hitgroup_records;

    // Size info to track allocations
    size_t m_hitgroup_record_size;
    size_t m_hitgroup_record_count;

    // Keep previous data for cleanup
    size_t m_prev_hitgroup_record_count;

    // Helper methods
    void cleanupHitgroupRecords();
    bool allocateHitgroupRecords(size_t count);
    bool populateHitgroupRecord(HitGroupSbtRecord& record, std::shared_ptr<GeometryInstance> instance);
    
    // Utility for setting material flags
    void configureTexturesForMaterial(HitGroupSbtRecord& record, MaterialResource* matResource, size_t instanceIdx);
};

} // namespace optix_renderer