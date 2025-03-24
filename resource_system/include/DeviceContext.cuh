#pragma once

#include <unordered_map>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "ResourceTypes.h"
#include <iostream>
#include <iomanip>
#include "../../sutil/Exception.h"
#include "../../sutil/Preprocessor.h"
#include "GeometryResource.h"
#include "TextureResource.h"
#include "MaterialResource.h"
// Error checking macro (optional)
#define CUDA_DRIVER_CHECK(call)                                  \
    do {                                                         \
        CUresult err = call;                                     \
        if (err != CUDA_SUCCESS) {                               \
            const char *errStr;                                  \
            cuGetErrorString(err, &errStr);                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Driver Error: " << errStr << "\n";\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
} while(0)
namespace optix_renderer {

// Forward declarations
//class GeometryResource;
//class TextureResource;
//class MaterialResource;

// Device-specific geometry data
struct DeviceGeometryData {
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;
    CUdeviceptr d_normals = 0;
    CUdeviceptr d_texCoords = 0;
    CUdeviceptr d_tangents = 0;
    CUdeviceptr d_bitangents = 0;
    OptixTraversableHandle traversable = 0;
    CUdeviceptr d_gas_output_buffer = 0;
    size_t gas_output_buffer_size = 0;

    OptixAccelEmitDesc emitProperty = {};
    
    // Cleanup
    void release();
};

// Device-specific texture data
struct DeviceTextureData {
    CUtexObject texture_object = 0;
    CUarray array = 0;
    CUmipmappedArray mipmapped_array = 0;
    size_t bytes_allocated = 0;
    int original_channels = 0;  // Original number of channels before padding (e.g. 3 for RGB)
    bool is_half_precision = false;   // True if texture is using half precision (fp16)
    bool is_int8_precision = false;   // True if texture is using int8/uint8 precision
    int original_data_type = 0;       // Original torch::ScalarType value
    
    // Cleanup
    void release();
};

// Device-specific material data
struct DeviceMaterialData {
    CUdeviceptr d_parameters = 0;
    size_t parameters_size = 0;
    std::vector<CUtexObject> texture_objects;
    
    // Cleanup
    void release();
};
static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}
// Manages resources on a specific GPU device
class DeviceContext {
public:
    DeviceContext(int deviceId = 0);
    ~DeviceContext(){}

    // Initialize and cleanup
    bool initialize();
    void destroy(){}

    // Access
    int getDeviceId() const { return m_deviceId; }
    CUstream getStream() const { return m_stream; }
    OptixDeviceContext getContext() const { return m_optixContext; }
    bool isInitialized() const { return m_initialized; }
    
    // Resource allocation on this device
    bool allocateGeometry(ResourceID id, GeometryResource* geometry);
    bool allocateTexture(ResourceID id, TextureResource* texture);
    bool allocateMaterial(ResourceID id, MaterialResource* material);
    
    // Resource deallocation
    void releaseGeometry(ResourceID id){}
    void releaseTexture(ResourceID id){}
    void releaseMaterial(ResourceID id){}
    
    // Get device-specific data
    DeviceGeometryData* getGeometryData(ResourceID id){
        auto it = m_geometryData.find(id);
        if (it != m_geometryData.end()){
            return &(it->second);
        }
        return nullptr;
    }
    DeviceTextureData* getTextureData(ResourceID id){
        auto it = m_textureData.find(id);
        if (it != m_textureData.end()){
            return &(it->second);
        }
        return nullptr;
    }
    DeviceMaterialData* getMaterialData(ResourceID id){
        auto it = m_materialData.find(id);
        if (it != m_materialData.end()){
            return &(it->second);
        }
        return nullptr;
    }
    
    // Synchronization
    void synchronize(){}
    
private:
    unsigned int triangle_flags = OPTIX_GEOMETRY_FLAG_NONE;//OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    // Build acceleration structure for geometry
    bool buildAccelerationStructure(ResourceID id, GeometryResource* geometry, DeviceGeometryData* deviceData);
    
    // Create CUDA texture objects
    bool createTextureObject(ResourceID id, TextureResource* texture, DeviceTextureData* deviceData);
    
public:
    // Set device as current - should be called before any CUDA or OptiX operations
    bool setDevice() const {
        try {
            // First use CUDA runtime API to ensure we're on the right device
            cudaSetDevice(m_deviceId);
            
            // Then use CUDA driver API to set the current device context
            CUDA_DRIVER_CHECK(cuCtxSetCurrent(NULL)); // Clear current context first
            
            // Select our device explicitly
            CUdevice device;
            CUDA_DRIVER_CHECK(cuDeviceGet(&device, m_deviceId));
            
            // Push our context on top of the context stack
            CUDA_DRIVER_CHECK(cuCtxPushCurrent(m_cuContext));
            
            // Only get device name in debug builds
            #ifdef DEBUG
            char deviceName[256];
            CUDA_DRIVER_CHECK(cuDeviceGetName(deviceName, 256, device));
            std::cout << "GPU " << m_deviceId << ": Active" << std::endl;
            #endif
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error setting device " << m_deviceId << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    // Pop the context when finished with operations
    void releaseDevice() const {
        CUcontext popped;
        //CUDA_DRIVER_CHECK(cuCtxPopCurrent(&popped));
    }
private:
    
    // Module loading and PTX compilation
public:
    OptixModule loadOptixModule(const std::string& ptxCode, const OptixModuleCompileOptions& moduleCompileOptions);
    OptixProgramGroup createProgramGroup(OptixModule module, const std::string& raygenName, 
                                       const std::string& missName, const std::string& hitGroupName);
    std::string       loadPTXFile(const std::string& filename);


private:
    int m_deviceId;
    bool m_initialized;
    
    // CUDA resources
    CUstream m_stream;
    // CUDA device handle
    CUdevice m_cuDevice;
    // CUcontext device contxt
    CUcontext m_cuContext;
    
    // OptiX resources
    OptixDeviceContext m_optixContext;
    
    // Compiled modules and program groups
    std::unordered_map<std::string, OptixModule> m_modules;
    std::unordered_map<std::string, OptixProgramGroup> m_programGroups;
    
    // Device-specific resource data
    std::unordered_map<ResourceID, DeviceGeometryData> m_geometryData;
    std::unordered_map<ResourceID, DeviceTextureData> m_textureData;
    std::unordered_map<ResourceID, DeviceMaterialData> m_materialData;
};

} // namespace optix_renderer