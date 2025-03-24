#include "../include/DeviceContext.cuh"
#include "../../sutil/Exception.h"
#include <vector_types.h>

// For making fake stub implementations during development

namespace optix_renderer {

    DeviceContext::DeviceContext(int deviceId) { m_deviceId = deviceId; }

    bool DeviceContext::initialize() {
        // Set the CUDA device using the runtime API
        cudaSetDevice(m_deviceId);
        
        // Initialize the CUDA driver API
        CUDA_DRIVER_CHECK(cuInit(0));
        
        // Get the device handle
        CUDA_DRIVER_CHECK(cuDeviceGet(&m_cuDevice, m_deviceId));
        
        // Create a primary context for the device with flags for optimal performance
        CUDA_DRIVER_CHECK(cuCtxCreate(&m_cuContext, CU_CTX_SCHED_AUTO, m_cuDevice));
        
        // Set our context as current
        CUDA_DRIVER_CHECK(cuCtxSetCurrent(m_cuContext));
        
        // Create a stream for asynchronous operations
        CUDA_DRIVER_CHECK(cuStreamCreate(&m_stream, CU_STREAM_DEFAULT));


        // Now cuContext is ready to use
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        //#ifdef DEBUG
        // This may incur significant performance cost and should only be done during development.
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        //#endif
        OPTIX_CHECK(optixDeviceContextCreate(m_cuContext, &options, &m_optixContext));

        // init complete
        m_initialized = true;
        return true;
    }

    bool DeviceContext::allocateGeometry(ResourceID id, GeometryResource *geometry) {
        // 1. Create device memory for vertices, indices, etc.
        // 2. Upload geometry data to the device
        // 3. Build acceleration structure if needed
        // 4. Store in the geometry data map

        DeviceGeometryData data;
        // Allocate and copy vertex data
        size_t vertexBytes = geometry->getVertices().numel() * sizeof(float);
        if (!setDevice()) return false;
        std::cout << "allocating geometry data for: " << m_deviceId << std::endl;

        CUDA_DRIVER_CHECK(cuMemAllocAsync(&data.d_vertices, vertexBytes, m_stream));
        CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(
            data.d_vertices,
            geometry->getVertices().data_ptr(),
            vertexBytes,
            m_stream
        ));


        if (geometry->getIndices().has_value()) {
            size_t indicesBytes = geometry->getIndices()->numel() * sizeof(uint32_t);
            CUDA_DRIVER_CHECK(cuMemAllocAsync(&data.d_indices, indicesBytes, m_stream));
            CUDA_DRIVER_CHECK(cuMemcpyHtoD(
                               data.d_indices,
                               geometry->getIndices()->data_ptr(),
                               indicesBytes));
        }

        if (geometry->getNormals().has_value()) {
            size_t normalBytes = geometry->getNormals()->numel() * sizeof(float3);
            CUDA_DRIVER_CHECK(cuMemAllocAsync(&data.d_normals, normalBytes, m_stream));
            CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(
                   data.d_normals,
                   geometry->getNormals()->data_ptr(),
                   normalBytes, m_stream));
        }

        if (geometry->getTexCoords().has_value()) {
            size_t texCoordsBytes = geometry->getTexCoords()->numel() * sizeof(float2);
            CUDA_DRIVER_CHECK(cuMemAllocAsync(&data.d_texCoords, texCoordsBytes, m_stream));
            CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(
                   data.d_texCoords,
                   geometry->getTexCoords()->data_ptr(),
                   texCoordsBytes, m_stream));
        }

        if (geometry->getTangents().has_value()) {
            size_t tangentBytes = geometry->getTangents()->numel() * sizeof(float3);
            CUDA_DRIVER_CHECK(cuMemAllocAsync(&data.d_tangents, tangentBytes, m_stream));
            CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(
                   data.d_tangents,
                   geometry->getTangents()->data_ptr(),
                   tangentBytes, m_stream));
        }
        if (geometry->getBitangents().has_value()) {
            size_t bitangentBytes = geometry->getBitangents()->numel() * sizeof(float3);
            CUDA_DRIVER_CHECK(cuMemAllocAsync(&data.d_bitangents, bitangentBytes, m_stream));
            CUDA_DRIVER_CHECK(cuMemcpyHtoDAsync(
                   data.d_bitangents,
                   geometry->getBitangents()->data_ptr(),
                   bitangentBytes, m_stream));
        }


        // Similar process for indices, normals, etc.

        // Build acceleration structure
        buildAccelerationStructure(id, geometry, &data);

        // Store the data
        m_geometryData[id] = data;
        return true;
    }
    template<typename IntegerType>
    SUTIL_INLINE SUTIL_HOSTDEVICE IntegerType roundUp(IntegerType x, IntegerType y) {
        return ((x + y - 1) / y) * y;
    }
    bool DeviceContext::buildAccelerationStructure(ResourceID id, GeometryResource* geometry, DeviceGeometryData* deviceData){
        // Print debug information
        std::cout << "Building acceleration structure for geometry ID: " << id << std::endl;
        std::cout << "Number of vertices: " << geometry->getVertices().size(0) << std::endl;
        if (geometry->getIndices().has_value()) {
            std::cout << "Number of index triplets: " << geometry->getIndices()->size(0) << std::endl;
        } else {
            std::cout << "No indices provided" << std::endl;
        }
        
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
        build_input.triangleArray.numVertices = static_cast<unsigned int>(geometry->getVertices().size(0));
        build_input.triangleArray.vertexBuffers = &deviceData->d_vertices;

        if (geometry->getIndices().has_value()) {
            build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
            build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(geometry->getIndices()->size(0));
            build_input.triangleArray.indexBuffer = deviceData->d_indices;
        }

        build_input.triangleArray.flags = &triangle_flags;
        build_input.triangleArray.numSbtRecords = 1;
        build_input.triangleArray.sbtIndexOffsetBuffer = 0;
        build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
        build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes = {};

        // Compute memory usage
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
                m_optixContext,
                &accel_options,
                &build_input,
                1,
                &gas_buffer_sizes));

    // Allocate temporary buffers
    CUdeviceptr d_temp_buffer = 0;

    // Only allocate if we need temp space
    if (gas_buffer_sizes.tempSizeInBytes > 0) {
        size_t temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
        CUDA_DRIVER_CHECK(cuMemAllocAsync(&d_temp_buffer, temp_buffer_size, m_stream));
    }

    // Allocate output buffer
    CUDA_DRIVER_CHECK(cuMemAllocAsync(&deviceData->d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, m_stream));
    deviceData->gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;

    CUdeviceptr &d_buffer_temp_output_gas_and_compacted_size = deviceData->d_gas_output_buffer;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);

    CUDA_DRIVER_CHECK(cuMemAllocAsync(
            &d_buffer_temp_output_gas_and_compacted_size,
            compactedSizeOffset + 8, m_stream));

    OptixAccelEmitDesc &emitProperty = deviceData->emitProperty;
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr) ((char *) d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // Build the GAS
    //OptixTraversableHandle gas_handle;
    OPTIX_CHECK(optixAccelBuild(
            m_optixContext,
            m_stream,// CUDA stream
            &accel_options,
            &build_input,
            1,// num build inputs
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size,
            gas_buffer_sizes.outputSizeInBytes,
            &deviceData->traversable,
            &emitProperty,// emitted property list
            1             // num emitted properties
            ));
            
    // Add debug printing to help diagnose the issue
    std::cout << "Built GAS for geometry ID: " << id << " with traversable handle: " << deviceData->traversable << std::endl;
    
    // Verify traversable handle is valid
    if (deviceData->traversable == 0) {
        std::cerr << "ERROR: Built GAS handle is 0 (invalid)!" << std::endl;
    }
            
    return true;

    }

    bool DeviceContext::allocateTexture(ResourceID id, TextureResource *texture) {
        // 1. Create CUDA array for the texture
        // 2. Upload texture data
        // 3. Create texture object
        // 4. Store in the texture data map

        DeviceTextureData data;
        
        std::cout << "Allocating texture for ID " << id << ":" << std::endl;
        std::cout << "  Tensor shape: " << texture->getData().sizes() << std::endl;
        std::cout << "  Data type: " << torch::toString(texture->getData().dtype()) << std::endl;

        // CUDA arrays support 1, 2, and 4 channels but not 3 channels directly
        // Get the number of channels, ensuring it's CUDA-compatible (1, 2, or 4)
        int channels = texture->getChannels();
        if (channels == 3) {
            // For RGB textures, we use 4 channels (RGBA) in CUDA arrays
            channels = 4;
            std::cout << "  Converting RGB texture to RGBA for CUDA array" << std::endl;
        }
        
        // Setup array descriptor
        CUDA_ARRAY_DESCRIPTOR arrayDesc;
        arrayDesc.Width = texture->getWidth();
        arrayDesc.Height = texture->getHeight();
        
        // Choose appropriate CUDA format based on original data type
        if (texture->isHalfPrecision()) {
            // Store the data in half precision (fp16) if original was half
            arrayDesc.Format = CU_AD_FORMAT_HALF;
            data.is_half_precision = true;
            std::cout << "  Using HALF precision format for texture" << std::endl;
        } else if (texture->isInt8Precision()) {
            // Store the data in 8-bit format if original was uint8/int8
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            data.is_int8_precision = true;
            std::cout << "  Using UNSIGNED_INT8 format for texture" << std::endl;
        } else {
            // Default to float format
            arrayDesc.Format = CU_AD_FORMAT_FLOAT;
            std::cout << "  Using FLOAT format for texture" << std::endl;
        }
        
        arrayDesc.NumChannels = channels;
        std::cout << "  CUDA array: " << arrayDesc.Width << "x" << arrayDesc.Height 
                 << " with " << arrayDesc.NumChannels << " channels" << std::endl;

        // Create array and copy data
        if (!setDevice()) return false;
        
        // Create the CUDA array
        CUDA_DRIVER_CHECK(cuArrayCreate(&data.array, &arrayDesc));
        
        // Get tensor data and make sure it's contiguous for reliable memory layout
        torch::Tensor tensor_data = texture->getData();
        if (!tensor_data.is_contiguous()) {
            std::cout << "  Data tensor is not contiguous, making contiguous copy" << std::endl;
            tensor_data = tensor_data.contiguous();
        }
        
        // Copy data, handling the case where input is RGB but CUDA array is RGBA
        CUDA_MEMCPY2D copyParams = {};
        copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
        copyParams.srcHost = tensor_data.data_ptr();
        
        // Source pitch depends on the original texture's channels
        size_t element_size = sizeof(float); // Default for float32
        if (texture->isHalfPrecision()) {
            // Half precision (fp16) is 2 bytes per element
            element_size = sizeof(uint16_t);
        } else if (texture->isInt8Precision()) {
            // int8/uint8 is 1 byte per element
            element_size = sizeof(uint8_t);
        }
        
        // Set source and destination layout information
        copyParams.srcPitch = texture->getWidth() * texture->getChannels() * element_size;
        copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyParams.dstArray = data.array;
        
        // Width is the minimum of source and destination channels
        int copy_channels = std::min(texture->getChannels(), channels);
        copyParams.WidthInBytes = texture->getWidth() * copy_channels * element_size;
        copyParams.Height = texture->getHeight();
        
        std::cout << "  Copy params - srcPitch: " << copyParams.srcPitch 
                 << ", WidthInBytes: " << copyParams.WidthInBytes 
                 << ", Height: " << copyParams.Height << std::endl;

        // Perform the copy
        CUDA_DRIVER_CHECK(cuMemcpy2D(&copyParams));
        std::cout << "  Data copy to CUDA array completed successfully" << std::endl;

        // Store information about original format for proper sampling in shaders
        data.original_channels = texture->getOriginalChannels();
        data.is_half_precision = texture->isHalfPrecision();
        data.is_int8_precision = texture->isInt8Precision();
        data.original_data_type = static_cast<int>(texture->getOriginalDataType());
        
        // Create texture object
        createTextureObject(id, texture, &data);

        // Store the data
        m_textureData[id] = data;
        releaseDevice();
        return true;
    }

    bool DeviceContext::createTextureObject(ResourceID id, TextureResource* texture, DeviceTextureData* deviceData) {

        // Create a CUDA resource descriptor
        CUDA_RESOURCE_DESC resDesc = {};
        resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
        resDesc.res.array.hArray = deviceData->array;

        // Create texture descriptor
        CUDA_TEXTURE_DESC texDesc = {};
        texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP; // Use WRAP for better tiling
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
        texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR; // Use LINEAR for smoother sampling
        
        // Make sure texture coordinates are normalized (important!)
        texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
        
        // Set read mode based on texture data type
        if (texture->isInt8Precision()) {
            // For 8-bit data, normalize to [0,1] range
            texDesc.flags |= CU_TRSF_READ_AS_INTEGER;
        } else {
            // For floating point data, use as-is
            // Nothing to set - default is float mode
        }
        
        // Store the original channel count to help the shader access the texture correctly
        deviceData->original_channels = texture->getOriginalChannels();

        // Print debug info about the texture
        std::cout << "Creating texture object for ID " << id << ":" << std::endl;
        std::cout << "  Size: " << texture->getWidth() << "x" << texture->getHeight() << std::endl;
        std::cout << "  Channels: " << texture->getChannels() << std::endl;
        std::cout << "  Is half precision: " << (texture->isHalfPrecision() ? "yes" : "no") << std::endl;
        std::cout << "  Is int8 precision: " << (texture->isInt8Precision() ? "yes" : "no") << std::endl;
        std::cout << "  Texture flags: " << texDesc.flags << std::endl;
        
        CUDA_DRIVER_CHECK(cuTexObjectCreate(&deviceData->texture_object, &resDesc, &texDesc, nullptr ));
        std::cout << "  Created texture object: " << deviceData->texture_object << std::endl;

        return true;
    }

    bool DeviceContext::allocateMaterial(ResourceID id, MaterialResource *material) {
        // For a single OptiX shader implementation, we'll use a simplified approach
        // that maps all material types to a unified structure
        if (!setDevice()) return false;
        
        DeviceMaterialData data;
        
        // Simple material struct that matches what's expected by our OptiX shader
        struct SimpleMaterial {
            float3 albedo;          // Base color for all material types
            float3 emission;        // Emission color and strength (if emissive)
            float metallic;         // 0 = dielectric, 1 = metal
            float roughness;        // 0 = smooth, 1 = rough
            float transmission;     // 0 = opaque, 1 = transparent
            float ior;              // Index of refraction for transparent materials
            int albedoTexture;      // Texture handle index or -1 if not used
            int materialType;       // Corresponds to MaterialType enum
        };
        
        // Create a material suitable for our OptiX shader
        SimpleMaterial gpuMat = {};
        
        // Initialize with default values
        gpuMat.albedo = make_float3(0.8f, 0.8f, 0.8f);
        gpuMat.emission = make_float3(0.0f, 0.0f, 0.0f);
        gpuMat.metallic = 0.0f;
        gpuMat.roughness = 0.5f;
        gpuMat.transmission = 0.0f;
        gpuMat.ior = 1.5f;
        gpuMat.albedoTexture = -1;
        gpuMat.materialType = static_cast<int>(material->getMaterialType());
        
        // Fill in material-specific parameters
        switch (material->getMaterialType()) {
            case MaterialType::LAMBERTIAN:
                if (material->hasParameter("albedo")) {
                    MaterialParameter param = material->getParameter("albedo");
                    if (param.getType() == ParameterType::FLOAT3) {
                        gpuMat.albedo = param.asFloat3();
                    }
                }
                break;
                
            case MaterialType::PBR:
                if (material->hasParameter("base_color")) {
                    MaterialParameter param = material->getParameter("base_color");
                    if (param.getType() == ParameterType::FLOAT3) {
                        gpuMat.albedo = param.asFloat3();
                    }
                }
                if (material->hasParameter("metallic")) {
                    MaterialParameter param = material->getParameter("metallic");
                    gpuMat.metallic = param.asFloat();
                }
                if (material->hasParameter("roughness")) {
                    MaterialParameter param = material->getParameter("roughness");
                    gpuMat.roughness = param.asFloat();
                }
                break;
                
            case MaterialType::EMISSIVE:
                if (material->hasParameter("emission_color")) {
                    MaterialParameter param = material->getParameter("emission_color");
                    if (param.getType() == ParameterType::FLOAT3) {
                        gpuMat.emission = param.asFloat3();
                    }
                }
                if (material->hasParameter("emission_strength")) {
                    MaterialParameter param = material->getParameter("emission_strength");
                    float strength = param.asFloat();
                    // Scale emission by strength
                    gpuMat.emission.x *= strength;
                    gpuMat.emission.y *= strength;
                    gpuMat.emission.z *= strength;
                }
                break;
                
            case MaterialType::GLASS:
                if (material->hasParameter("ior")) {
                    MaterialParameter param = material->getParameter("ior");
                    gpuMat.ior = param.asFloat();
                }
                gpuMat.transmission = 1.0f;  // Glass is transparent by default
                if (material->hasParameter("roughness")) {
                    MaterialParameter param = material->getParameter("roughness");
                    gpuMat.roughness = param.asFloat();
                } else {
                    gpuMat.roughness = 0.0f;  // Perfect glass is smooth
                }
                break;
                
            case MaterialType::MIRROR:
                if (material->hasParameter("tint")) {
                    MaterialParameter param = material->getParameter("tint");
                    if (param.getType() == ParameterType::FLOAT3) {
                        gpuMat.albedo = param.asFloat3();
                    }
                }
                gpuMat.metallic = 1.0f;      // Mirrors are perfectly metallic
                gpuMat.roughness = 0.0f;     // Mirrors are perfectly smooth
                break;
                
            default:
                // Unknown material type, use defaults
                break;
        }
        
        // Handle textures - for simplicity we'll just support an albedo texture for now
        const std::vector<std::string> textureNames = {
            "albedo", "albedo_texture", "base_color_texture"
        };
        
        for (const auto& texName : textureNames) {
            if (material->hasTexture(texName)) {
                TextureHandle texHandle = material->getTexture(texName);
                auto texIt = m_textureData.find(texHandle);
                
                if (texIt != m_textureData.end()) {
                    // Add the texture object to our list
                    data.texture_objects.push_back(texIt->second.texture_object);
                    
                    // Set the texture index
                    gpuMat.albedoTexture = static_cast<int>(texHandle);
                    break;  // Just use the first texture we find
                }
            }
        }
        
        // Allocate and upload the material data
        data.parameters_size = sizeof(SimpleMaterial);
        CUDA_DRIVER_CHECK(cuMemAlloc(&data.d_parameters, data.parameters_size));
        CUDA_DRIVER_CHECK(cuMemcpyHtoD(data.d_parameters, &gpuMat, data.parameters_size));
        
        // Store the data
        m_materialData[id] = data;
        return true;
    }

    // Read a PTX file from disk
    std::string DeviceContext::loadPTXFile(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Error opening PTX file: " + filename);
        }

        // Get the file size
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read the file content
        std::string content(size, ' ');
        file.read(&content[0], size);

        if (!file) {
            throw std::runtime_error("Error reading PTX file: " + filename);
        }

        return content;
    }

    // Load and compile an OptiX module from PTX code
    OptixModule DeviceContext::loadOptixModule(const std::string &ptxCode,
                                               const OptixModuleCompileOptions &moduleCompileOptions) {
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipelineCompileOptions.numPayloadValues = 3;
        pipelineCompileOptions.numAttributeValues = 3;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

        OptixModule module;
        char log[2048];
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK(optixModuleCreate(
            m_optixContext,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            ptxCode.c_str(),
            ptxCode.size(),
            log,
            &sizeof_log,
            &module
        ));

        if (sizeof_log > 1) {
            std::cout << "Module compilation log: " << log << std::endl;
        }

        return module;
    }

    // Create program groups for ray generation, miss, and hit programs
    OptixProgramGroup DeviceContext::createProgramGroup(OptixModule module,
                                                        const std::string &raygenName,
                                                        const std::string &missName,
                                                        const std::string &hitGroupName) {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};

        // Setup for ray generation program
        if (!raygenName.empty()) {
            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            pgDesc.raygen.module = module;
            pgDesc.raygen.entryFunctionName = raygenName.c_str();
        }

        // Setup for miss program
        if (!missName.empty()) {
            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            pgDesc.miss.module = module;
            pgDesc.miss.entryFunctionName = missName.c_str();
        }

        // Setup for hit group program
        if (!hitGroupName.empty()) {
            pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            pgDesc.hitgroup.moduleCH = module;
            pgDesc.hitgroup.entryFunctionNameCH = hitGroupName.c_str();
        }

        OptixProgramGroup pg;
        char log[2048];
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK(optixProgramGroupCreate(
            m_optixContext,
            &pgDesc,
            1, // num program groups
            &pgOptions,
            log,
            &sizeof_log,
            &pg
        ));

        if (sizeof_log > 1) {
            std::cout << "Program group compilation log: " << log << std::endl;
        }

        return pg;
    }

}// namespace optix_renderer