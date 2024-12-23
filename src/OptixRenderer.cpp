//
// Created by Artyom on 12/3/2024.
//

#include "OptixRenderer.h"

const int32_t g_tessellation_resolution = 128;

const float g_exploding_gas_rebuild_frequency = 10.f;

const int32_t INST_COUNT = 5;

const std::array<float3, INST_COUNT> g_diffuse_colors =
        {{
                {0.1f, 0.0f, 0.0f},
                {0.0f, 0.1f, 0.0f},
                {0.90f, 0.90f, 0.90f},
                {1.00f, 1.00f, 1.00f},
                {0.80f, 0.80f, 0.80f},
        }};

struct Instance {
    float m[12];
};

const std::array<Instance, INST_COUNT> g_instances =
        {{{{1, 0, 0, -4.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
          {{1, 0, 0, -1.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
          {{1, 0, 0, 1.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
          {{1, 0, 0, 4.5f, 0, 1, 0, 0, 0, 0, 1, 0}},
          {{1, 0, 0, 0.0f, 0, 1, 0, 0, 0, 0, 1, 0}}}};


OptixRenderer::~OptixRenderer() {
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(hit_group));
    OPTIX_CHECK(optixModuleDestroy(ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_vertices)));
    //CUDA_CHECK( cudaFree( reinterpret_cast< void* >(  d_static_gas_output_buffer ) ) );
    //CUDA_CHECK( cudaFree( reinterpret_cast< void* >(  d_deforming_gas_output_buffer ) ) );
    //CUDA_CHECK( cudaFree( reinterpret_cast< void* >(  d_exploding_gas_output_buffer ) ) );
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_instances)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_ias_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
}
static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}
void OptixRenderer::createContext() {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext cu_ctx = 0;// zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
//#ifdef DEBUG
    // This may incur significant performance cost and should only be done during development.
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
//#endif
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    this->context = context;
}
void OptixRenderer::createModule() {
    OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    this->pipeline_compile_options.usesMotionBlur = false;
    this->pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    this->pipeline_compile_options.numPayloadValues = 4;
    this->pipeline_compile_options.numAttributeValues = 3;
    this->pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    this->pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    this->pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;


    const char *ptx_filename = "kernel.ptx";
    std::ifstream ptx_file(ptx_filename);
    if (!ptx_file.is_open()) {
        std::cerr << "Failed to open PTX file: " << ptx_filename << std::endl;
        exit(1);
    }
    std::string ptx_source((std::istreambuf_iterator<char>(ptx_file)), std::istreambuf_iterator<char>());
    ptx_file.close();

    OPTIX_CHECK_LOG(optixModuleCreate(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            ptx_source.c_str(),
            ptx_source.size(),
            LOG, &LOG_SIZE,
            &ptx_module));
}
void OptixRenderer::createProgramGroups() {
    OptixProgramGroupOptions program_group_options = {};

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context, &raygen_prog_group_desc,
                1,// num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &raygen_prog_group));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context, &miss_prog_group_desc,
                1,// num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &miss_group));
    }

    {
        // Create program group for any-hit shader
//        OptixProgramGroupDesc any_hit_desc = {};
//        any_hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
//        any_hit_desc.hitgroup.moduleAH = ptx_module; // Your any-hit program group
//        any_hit_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
//
//        OPTIX_CHECK_LOG(optixProgramGroupCreate(
//                context, &any_hit_desc,
//                1,
//                &program_group_options,
//                LOG, &LOG_SIZE,
//                &any_hit_prog_group));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        hit_prog_group_desc.hitgroup.moduleAH = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &hit_prog_group_desc,
                1,// num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &hit_group));
    }
}
void OptixRenderer::createPipeline() {
    OptixProgramGroup program_groups[] =
            {
                    raygen_prog_group,
                    miss_group,
                    hit_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 10;

    OPTIX_CHECK_LOG(optixPipelineCreate(
            context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            LOG, &LOG_SIZE,
            &pipeline));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(raygen_prog_group, &stack_sizes, pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(miss_group, &stack_sizes, pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(hit_group, &stack_sizes, pipeline));
    //OPTIX_CHECK(optixUtilAccumulateStackSizes(any_hit_prog_group, &stack_sizes, pipeline));

    uint32_t max_trace_depth = 10;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
            &stack_sizes,
            max_trace_depth,
            max_cc_depth,
            max_dc_depth,
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state,
            &continuation_stack_size));

    // This is 2 since the largest depth is IAS->GAS
    const uint32_t max_traversable_graph_depth = 2;

    OPTIX_CHECK(optixPipelineSetStackSize(
            pipeline,
            direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state,
            continuation_stack_size,
            max_traversable_graph_depth));
}
void OptixRenderer::createSBT() {
    CUdeviceptr d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(d_raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));

    CUdeviceptr d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss_records), miss_record_size));

    MissRecord ms_sbt[1];

    OPTIX_CHECK(optixSbtRecordPackHeader(miss_group, &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.4f, 0.3f, 0.3f, 0.f);

    CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(d_miss_records),
            ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice));

    initSBT(16);

    //std::vector<HitGroupRecord> hitgroup_records(0);
    //for (int i = 0; i < static_cast<int>(g_instances.size()); ++i) {
    //    const int sbt_idx = i;
//
    //    OPTIX_CHECK(optixSbtRecordPackHeader(hit_group, &hitgroup_records[sbt_idx]));
    //    hitgroup_records[sbt_idx].data.color = g_diffuse_colors[i];
    //}

    //CUDA_CHECK(cudaMemcpy(
    //        reinterpret_cast<void *>(sbt_data.d_sbt_records),
    //        hitgroup_records.data(),
    //        sbt_data.record_size * hitgroup_records.size(),
    //        cudaMemcpyHostToDevice));

    //sbt_data.count += hitgroup_records.size();
    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_records;
    sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = sbt_data.d_sbt_records;
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sbt_data.record_size);
    sbt.hitgroupRecordCount = static_cast<uint32_t>(sbt_data.count);
}
void OptixRenderer::initLaunchParams() {
    CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>(&params.accum_buffer),
            params.width * params.height * sizeof( float4 )));
    params.frame_buffer = nullptr;// Will be set when output buffer is mapped
    params.subframe_index = 0u;
    params.samples_per_launch = 1u;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
}
template<typename IntegerType>
SUTIL_INLINE SUTIL_HOSTDEVICE IntegerType roundUp(IntegerType x, IntegerType y) {
    return ((x + y - 1) / y) * y;
}
void OptixRenderer::launchGenerateAnimatedVertices(AnimationMode animation_mode, CUdeviceptr &d_temp_vertices) {
    generateAnimatedVetrices((float3 *) d_temp_vertices, animation_mode, time, g_tessellation_resolution, g_tessellation_resolution);
}

Geometry OptixRenderer::createVertexGeometry(torch::Tensor& vertices,
                                             torch::Tensor* indices,
                                             torch::Tensor* texCoords,
                                             torch::Tensor* tangents,
                                             torch::Tensor* bitangents,
                                             torch::Tensor* vertex_normals,
                                             TextureObject* texture,
                                             TextureObject* normals,
                                             TextureObject* metallic_roughness,
                                             TextureObject* emission_texture)
{
    CUdeviceptr d_vertices = reinterpret_cast<CUdeviceptr>(vertices.data_ptr());

    Geometry geometry = Geometry(*this, vertices, indices);

    geometry.build_input = {};
    geometry.build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    geometry.build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    geometry.build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    geometry.build_input.triangleArray.numVertices = static_cast<unsigned int>(vertices.size(0));
    geometry.build_input.triangleArray.vertexBuffers = &d_vertices;

    if(indices) {
        CUdeviceptr d_indices = reinterpret_cast<CUdeviceptr>(indices->data_ptr());
        geometry.build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        geometry.build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
        geometry.build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices->size(0));
        geometry.build_input.triangleArray.indexBuffer = d_indices;
        geometry.material.index_buffer = d_indices;
    }
    if(tangents){
        CUdeviceptr d_tangetnts = reinterpret_cast<CUdeviceptr>(tangents->data_ptr());
        geometry.material.tangents = d_tangetnts;
    }
    if(bitangents){
        CUdeviceptr d_bitangents = reinterpret_cast<CUdeviceptr>(bitangents->data_ptr());
        geometry.material.bitangents = d_bitangents;
    }
    if(vertex_normals){
        CUdeviceptr d_vertex_normals = reinterpret_cast<CUdeviceptr>(vertex_normals->data_ptr());
        geometry.material.normals = d_vertex_normals;
    }

    if (texCoords) {
        CUdeviceptr d_texCoords = reinterpret_cast<CUdeviceptr>(texCoords->data_ptr());
        geometry.texCoords = texCoords;
        geometry.material.uv_buffer = reinterpret_cast<CUdeviceptr>(texCoords->data_ptr());
    }
    if (texture) {
        geometry.material.texture = texture->get();
        geometry.material.texture_ptr = texture->get_img_ptr();
    }
    if(normals){
        geometry.material.normal_map = normals->get();
    }
    if(metallic_roughness){
        geometry.material.metallic_roughness = metallic_roughness->get();
    }
    if(emission_texture){
        geometry.material.emission_texture = emission_texture->get();
    }

    //uint32_t triangle_flags = OPTIX_GEOMETRY_FLAG_NONE;
    geometry.build_input.triangleArray.flags = &triangle_flags;
    geometry.build_input.triangleArray.numSbtRecords = 1;
    geometry.build_input.triangleArray.sbtIndexOffsetBuffer = 0;
    geometry.build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    geometry.build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Compute memory usage
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            &geometry.build_input,
            1,
            &gas_buffer_sizes));

    // Allocate temporary buffers
    //CUdeviceptr d_temp_buffer;
    //CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));
    if (gas_buffer_sizes.tempSizeInBytes > temp_buffer_size) {
        if (d_temp_buffer != 0) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
        }
        temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), temp_buffer_size));
    }

    // Allocate output buffer
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&geometry.d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));
    geometry.gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;

    CUdeviceptr &d_buffer_temp_output_gas_and_compacted_size = geometry.d_gas_output_buffer;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
            compactedSizeOffset + 8));

    OptixAccelEmitDesc &emitProperty = geometry.emitProperty;
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr) ((char *) d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);
    // Build the GAS
    //OptixTraversableHandle gas_handle;
    OPTIX_CHECK(optixAccelBuild(
            context,
            0,// CUDA stream
            &accel_options,
            &geometry.build_input,
            1,// num build inputs
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size,
            gas_buffer_sizes.outputSizeInBytes,
            &geometry.gas_handle,
            &emitProperty,// emitted property list
            1             // num emitted properties
            ));
    geometry.gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;
    // Clean up temporary buffer
    //CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    //temp_buffer_size = 0;

    return geometry;
}
torch::Tensor OptixRenderer::getTransformForInstance(uint32_t id) {
    torch::Tensor tensor = torch::from_blob(
         reinterpret_cast<void*>(((OptixInstance*)d_instances)[id].transform), // Raw CUDA pointer
        {3, 4},            // Sizes
        torch::TensorOptions()  // Tensor options
            .dtype(torch::kFloat32) // Data type
            .device(torch::kCUDA));
    return tensor;
}
std::ostream& operator<<(std::ostream& os, const OptixRenderer::SBTData& data) {
    os << "SBTData {"
       << "\n  d_sbt_records: 0x" << std::hex << data.d_sbt_records << std::dec
       << "\n  record_size: " << data.record_size
       << "\n  capacity: " << data.capacity
       << "\n  count: " << data.count
       << "\n}";
    return os;
}
std::ostream& operator<<(std::ostream& os, const HitGroupData& data) {
    os << "HitGroupData:\n"
       << "\n\tColor: (" << data.color.x << ", " << data.color.y << ", " << data.color.z << ")"
       << "\n\tIndex of Refraction (IOR): " << data.IOR
       << "\n\tIs Glass: " << (data.is_glass ? "Yes" : "No")
       << "\n\tTexture Object Handle: " << static_cast<unsigned long long>(data.texture)
       << "\n\tIndex Buffer Pointer: " << reinterpret_cast<void*>(data.index_buffer)
       << "\n\tUV Buffer Pointer: " << reinterpret_cast<void*>(data.uv_buffer)
       << "\n\ttexture Pointer: " << reinterpret_cast<void*>(data.texture_ptr);
    return os;
}
void OptixRenderer::addGeometryToSBT(Geometry& geometry){
    //std::cout<<"sbt befor add: ";
    //std::cout<<sbt_data<<std::endl;
    if (sbt_data.count >= sbt_data.capacity) {
        // Double the capacity
        size_t new_capacity = sbt_data.capacity * 2;
        CUdeviceptr new_sbt_records;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&new_sbt_records), sbt_data.record_size * new_capacity));

        // Copy existing records to the new buffer
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(new_sbt_records), reinterpret_cast<void*>(sbt_data.d_sbt_records),
                              sbt_data.record_size * sbt_data.capacity, cudaMemcpyDeviceToDevice));

        // Free the old buffer and update metadata
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_data.d_sbt_records)));
        sbt_data.d_sbt_records = new_sbt_records;
        sbt_data.capacity = new_capacity;
    }

    // Create the SBT record for this geometry
    HitGroupRecord hit_record = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hit_group, &hit_record));
    hit_record.data = geometry.material;
    //std::cout<<"imagedeskid: "<< hit_record.data.texture << " " << geometry.material.texture <<std::endl;
    //hit_record.data.print();
    // Print to check if buffers are correct
    //std::cout << "Index Buffer Pointer before GPU transfer: " << reinterpret_cast<void*>(hit_record.data.index_buffer) << std::endl;
    //std::cout << "UV Buffer Pointer before GPU transfer: " << reinterpret_cast<void*>(hit_record.data.uv_buffer) << std::endl;

    //std::cout<<"sbt used for copy: "<<sbt_data<<std::endl;
    // Copy the new record to the device
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt_data.d_sbt_records + sbt_data.count * sbt_data.record_size),
                          &hit_record, sbt_data.record_size, cudaMemcpyHostToDevice));
    //std::cout<<"cudaMemcpy(reinterpret_cast<void*>("<< std::hex <<sbt_data.d_sbt_records + sbt_data.count * sbt_data.record_size << std::dec<< ", "
              //<<&hit_record<<", "<< sbt_data.record_size<<") "<<std::endl;

    sbt_data.count++;

    //std::cout<<"sbt after add: "<<sbt_data<<std::endl;

    // Allocate a host buffer to hold the data
    std::vector<HitGroupRecord> host_buffer(sbt_data.count);

    // Copy data from the device to the host buffer
    CUDA_CHECK(cudaMemcpy(host_buffer.data(),
                          reinterpret_cast<void*>(sbt_data.d_sbt_records),
                          sbt_data.record_size * sbt_data.count,
                          cudaMemcpyDeviceToHost));
    //std::cout<<"first 4 on gpu"<<std::endl;
    //for(int i = 0; i < 4; i++){
    //    std::cout<<host_buffer[i].data<<"\n";
    //}
    //std::cout<<std::endl;
    updateSBT();
}
void OptixRenderer::initSBT(size_t initial_capacity){
    sbt_data.record_size = sizeof(HitGroupRecord);
    sbt_data.capacity = initial_capacity;
    sbt_data.count = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&sbt_data.d_sbt_records), sbt_data.record_size * sbt_data.capacity));
}
void OptixRenderer::updateSBT() {
    sbt.hitgroupRecordBase = sbt_data.d_sbt_records;
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sbt_data.record_size);
    sbt.hitgroupRecordCount = static_cast<uint32_t>(sbt_data.count);
}
uint32_t OptixRenderer::addGeometryInstance(Geometry &geometry) {
    OptixInstance instance = {};
    float transform[12] = {
            1, 0, 0, 0,// row 0
            0, 1, 0, 0,// row 1
            0, 0, 1, 0 // row 2
    };

    // Copy the transform matrix
    memcpy(instance.transform, transform, sizeof(float) * 12);

    // Set the cube's GAS handle
    instance.traversableHandle = geometry.gas_handle;

    // Set instance properties
    instance.instanceId = static_cast<unsigned int>(instances.size());// or any unique ID
    instance.sbtOffset = static_cast<unsigned int>(instances.size());
    instance.visibilityMask = 255;
    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    //geometry.instance = instance;
    // Add the cube instance to the instances vector
    instances.push_back(instance);

    addGeometryToSBT(geometry);
    updateSBT();
    return instances.size();
}

Geometry OptixRenderer::buidSphereGeometry() {
    // Allocate temporary space for vertex generation.
    // The same memory space is reused for generating the deformed and exploding vertices before updates.
    uint32_t numVertices = g_tessellation_resolution * g_tessellation_resolution * 6;


    torch::Device device(torch::kCUDA, 0);
    torch::Tensor vertices = torch::empty(
            {static_cast<long>(numVertices * 3)},
            torch::dtype(torch::kFloat32).device(device));
    d_temp_vertices = reinterpret_cast<CUdeviceptr>(vertices.data_ptr());
    launchGenerateAnimatedVertices(AnimationMode_None, d_temp_vertices);
    Geometry geometry = Geometry(*this, create_vertices_tensor(cube_vertices));
    // Build an AS over the triangles.
    // We use un-indexed triangles so we can explode the sphere per triangle.
    geometry.build_input = {};
    geometry.build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    geometry.build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    geometry.build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    geometry.build_input.triangleArray.numVertices = static_cast<uint32_t>(numVertices);
    geometry.build_input.triangleArray.vertexBuffers = &d_temp_vertices;
    geometry.build_input.triangleArray.flags = &triangle_flags;
    geometry.build_input.triangleArray.numSbtRecords = 1;
    geometry.build_input.triangleArray.sbtIndexOffsetBuffer = 0;
    geometry.build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    geometry.build_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            &geometry.build_input,
            1,// num_build_inputs
            &gas_buffer_sizes));

    //temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
    if (gas_buffer_sizes.tempSizeInBytes > temp_buffer_size) {
        if (d_temp_buffer != 0) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
        }
        temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), temp_buffer_size));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr &d_buffer_temp_output_gas_and_compacted_size = geometry.d_gas_output_buffer;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
            compactedSizeOffset + 8));

    OptixAccelEmitDesc &emitProperty = geometry.emitProperty;
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr) ((char *) d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
            context,
            0,// CUDA stream
            &accel_options,
            &geometry.build_input,
            1,// num build inputs
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size,
            gas_buffer_sizes.outputSizeInBytes,
            &geometry.gas_handle,
            &emitProperty,// emitted property list
            1             // num emitted properties
            ));
    geometry.gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;
    return geometry;
}
Geometry Geometry::copy() {
    Geometry geometry = Geometry(renderer, vertices, indices);
    geometry.build_input = build_input;
    geometry.gas_output_buffer_size = gas_output_buffer_size;
    geometry.emitProperty = emitProperty;
    geometry.material = material;

    OptixRelocationInfo relocationInfo;
    OPTIX_CHECK(optixAccelGetRelocationInfo(renderer.getContext(), gas_handle, &relocationInfo));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&geometry.d_gas_output_buffer), geometry.gas_output_buffer_size));
    CUDA_CHECK(cudaMemcpy((void *) geometry.d_gas_output_buffer, (const void *) d_gas_output_buffer, geometry.gas_output_buffer_size, cudaMemcpyDeviceToDevice));
    OPTIX_CHECK(optixAccelRelocate(renderer.getContext(), 0, &relocationInfo, 0, 0, geometry.d_gas_output_buffer, geometry.gas_output_buffer_size, &geometry.gas_handle));
    return geometry;
}
Geometry Geometry::compress() {
    // Compress GAS

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *) emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_output_buffer_size) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(renderer.getContext(), 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

        //            CUDA_CHECK( cudaFree( ( void* )d_buffer_temp_output_gas_and_compacted_size ) );

        gas_output_buffer_size = compacted_gas_size;
    }
    return *this;
}
void OptixRenderer::buildMeshAccel() {


    //Geometry geometry = buidSphereGeometry();
    //geometry.material.is_glass = 1;
    //geometry.material.IOR = 5.f;
    //geometry.material.color = make_float3(0.0f, 0.8f, 0.4f);
    //Geometry geometry_explosion = geometry.copy();
    // Replicate the uncompressed GAS for the exploding sphere.
    // The exploding sphere is occasionally rebuild. We don't want to compress the GAS after every rebuild so we use the uncompressed GAS for the exploding sphere.
    // The memory requirements for the uncompressed exploding GAS won't change so we can rebuild in-place.
    //exploding_gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;


    //Geometry geometry1 = createVertexGeometry(create_vertices_tensor(cube_vertices), create_indices_tensor(cube_indices));
    //addGeometryInstance(geometry1);

    //addGeometryInstance(geometry);
    //addGeometryInstance(geometry);
    //addGeometryInstance(geometry);


    //addGeometryInstance(geometry_explosion);


    //addGeometryInstance(geometry1.copy().compress());


    // Replicate the compressed GAS for the deforming sphere.
    // The deforming sphere is never rebuild so we refit the compressed GAS without requiring recompression.
    //deforming_gas_output_buffer_size =  static_gas_output_buffer_size;
    //CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( & d_deforming_gas_output_buffer ),  deforming_gas_output_buffer_size ) );
    //CUDA_CHECK( cudaMemcpy( ( void* ) d_deforming_gas_output_buffer, ( const void* ) d_static_gas_output_buffer,  deforming_gas_output_buffer_size, cudaMemcpyDeviceToDevice ) );
    //OPTIX_CHECK( optixAccelRelocate(  context, 0, &relocationInfo, 0, 0,  d_deforming_gas_output_buffer,  deforming_gas_output_buffer_size, & deforming_gas_handle ) );

    buidIAS();
}

void OptixRenderer::buidIAS() {
    // Build the IAS

    //for( size_t i = 0; i < g_instances.size(); ++i )
    //{
    //    memcpy( instances[i].transform, g_instances[i].m, sizeof( float ) * 12 );
    //    instances[i].sbtOffset = static_cast< unsigned int >( i );
    //    instances[i].visibilityMask = 255;
    //}

    //instances[0].traversableHandle =  static_gas_handle;
    //instances[1].traversableHandle =  static_gas_handle;
    //instances[2].traversableHandle =  deforming_gas_handle;
    //instances[3].traversableHandle =  exploding_gas_handle;
    //instances[4].traversableHandle =  static_gas_handle;


    //addGeometryInstance(geometry);
    //addGeometryInstance(geometry);
    //addGeometryInstance(geometry);
    //std::vector<OptixInstance> localInstances(this->instances.size());

    //for (int i = 0; i < this->instances.size(); i++) {
    //    localInstances[i] = this->instances[i];
    //}

    //addNewGeometryInstance(instances, geometry);


    size_t instances_size_in_bytes = sizeof(OptixInstance) * instances.size();
    CUDA_CHECK(cudaMalloc((void **) &d_instances, instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy((void *) d_instances, instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice));

    std::cout << "instances: " << instances.size() << std::endl;

    ias_instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ias_instance_input.instanceArray.instances = d_instances;
    ias_instance_input.instanceArray.numInstances = static_cast<int>(instances.size());

    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    ias_accel_options.motionOptions.numKeys = 1;
    ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &ias_accel_options, &ias_instance_input, 1, &ias_buffer_sizes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(ias_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_buffer_temp_output_ias_and_compacted_size), compactedSizeOffset + 8));

    CUdeviceptr d_ias_temp_buffer;
    bool needIASTempBuffer = ias_buffer_sizes.tempSizeInBytes > temp_buffer_size;
    if (needIASTempBuffer) {
        CUDA_CHECK(cudaMalloc((void **) &d_ias_temp_buffer, ias_buffer_sizes.tempSizeInBytes));
    } else {
        d_ias_temp_buffer = d_temp_buffer;
    }
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr) ((char *) d_buffer_temp_output_ias_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(context, 0, &ias_accel_options, &ias_instance_input, 1, d_ias_temp_buffer,
                                ias_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_ias_and_compacted_size,
                                ias_buffer_sizes.outputSizeInBytes, &ias_handle, &emitProperty, 1));

    if (needIASTempBuffer) {
        CUDA_CHECK(cudaFree((void *) d_ias_temp_buffer));
    }

    // Compress the IAS

    size_t compacted_ias_size;
    CUDA_CHECK(cudaMemcpy(&compacted_ias_size, (void *) emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_ias_size < ias_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ias_output_buffer), compacted_ias_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(context, 0, ias_handle, d_ias_output_buffer,
                                      compacted_ias_size, &ias_handle));

        CUDA_CHECK(cudaFree((void *) d_buffer_temp_output_ias_and_compacted_size));

        ias_output_buffer_size = compacted_ias_size;
    } else {
        d_ias_output_buffer = d_buffer_temp_output_ias_and_compacted_size;

        ias_output_buffer_size = ias_buffer_sizes.outputSizeInBytes;
    }

    // allocate enough temporary update space for updating the deforming GAS, exploding GAS and IAS.
    size_t maxUpdateTempSize = std::max(ias_buffer_sizes.tempUpdateSizeInBytes, gas_buffer_sizes.tempUpdateSizeInBytes);
    if (temp_buffer_size < maxUpdateTempSize) {
        CUDA_CHECK(cudaFree((void *) d_temp_buffer));
        temp_buffer_size = maxUpdateTempSize;
        CUDA_CHECK(cudaMalloc((void **) &d_temp_buffer, temp_buffer_size));
    }

    params.handle = ias_handle;
}

OptixRenderer::OptixRenderer(int width, int height) {
    params.width = width;
    params.height = height;
    createContext();
    createModule();
    createProgramGroups();
    createPipeline();
    createSBT();
    initLaunchParams();

    buildMeshAccel();
}
void OptixRenderer::updateState(torch::Tensor &output_tensor) {
    //handleCameraUpdate( params );
    //handleResize( output_tensor, params );
}
void printParams(const Params &params) {
    std::cout << "Params Debug Information:" << std::endl;
    std::cout << "  Frame Buffer: " << (params.frame_buffer ? "Set" : "Not Set") << std::endl;
    std::cout << "  Width: " << params.width << std::endl;
    std::cout << "  Height: " << params.height << std::endl;

    std::cout << "  Camera Eye: ("
              << params.eye.x << ", " << params.eye.y << ", " << params.eye.z << ")" << std::endl;
    std::cout << "  Camera U: ("
              << params.U.x << ", " << params.U.y << ", " << params.U.z << ")" << std::endl;
    std::cout << "  Camera V: ("
              << params.V.x << ", " << params.V.y << ", " << params.V.z << ")" << std::endl;
    std::cout << "  Camera W: ("
              << params.W.x << ", " << params.W.y << ", " << params.W.z << ")" << std::endl;

    std::cout << "  Traversable Handle: " << params.handle << std::endl;
    std::cout << "  Subframe Index: " << params.subframe_index << std::endl;
}

void OptixRenderer::launchSubframe(torch::Tensor &output_tensor) {
    // Launch
    //uchar4* result_buffer_data = output_buffer.map();
    uchar4 *result_buffer_data = reinterpret_cast<uchar4 *>(output_tensor.data_ptr<uint8_t>());
    params.frame_buffer = result_buffer_data;
    //printParams(params);
    CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(d_params),
            &params, sizeof(Params),
            cudaMemcpyHostToDevice, stream));

    OPTIX_CHECK(optixLaunch(
            pipeline,
            stream,
            reinterpret_cast<CUdeviceptr>(d_params),
            sizeof(Params),
            &sbt,
            params.width, // launch width
            params.height,// launch height
            1             // launch depth
            ));
    //output_buffer.unmap();
    CUDA_SYNC_CHECK();
}
#include <cmath>

// Function to set the rotation in the transform array
void setRotation(float X, float Y, float Z, float *transform) {
    // Convert angles from degrees to radians if necessary
    // Assuming X, Y, Z are in radians
    float cosX = cosf(X);
    float sinX = sinf(X);
    float cosY = cosf(Y);
    float sinY = sinf(Y);
    float cosZ = cosf(Z);
    float sinZ = sinf(Z);

    // Compute rotation matrices
    // Rotation around X-axis (Roll)
    float R_x[3][3] = {
            {1, 0, 0},
            {0, cosX, -sinX},
            {0, sinX, cosX}};

    // Rotation around Y-axis (Pitch)
    float R_y[3][3] = {
            {cosY, 0, sinY},
            {0, 1, 0},
            {-sinY, 0, cosY}};

    // Rotation around Z-axis (Yaw)
    float R_z[3][3] = {
            {cosZ, -sinZ, 0},
            {sinZ, cosZ, 0},
            {0, 0, 1}};

    // Combined rotation matrix R = R_z * R_y * R_x
    float R_zy[3][3];
    float R[3][3];

    // First multiply R_z and R_y: R_zy = R_z * R_y
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_zy[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
                R_zy[i][j] += R_z[i][k] * R_y[k][j];
            }
        }
    }

    // Then multiply R_zy and R_x: R = R_zy * R_x
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
                R[i][j] += R_zy[i][k] * R_x[k][j];
            }
        }
    }

    // Populate the transform array
    // Row 0
    transform[0] = R[0][0];
    transform[1] = R[0][1];
    transform[2] = R[0][2];
    // T₀ (translation along X-axis)
    // Leave as is or set to desired value
    // transform[3] = T0;

    // Row 1
    transform[4] = R[1][0];
    transform[5] = R[1][1];
    transform[6] = R[1][2];
    // T₁ (translation along Y-axis)
    // Leave as is or set to desired value
    // transform[7] = T1;

    // Row 2
    transform[8] = R[2][0];
    transform[9] = R[2][1];
    transform[10] = R[2][2];
    // T₂ (translation along Z-axis)
    // Leave as is or set to desired value
    // transform[11] = T2;
}


void OptixRenderer::updateMeshAccel() {
    /*
    // Generate deformed sphere vertices
    //    launchGenerateAnimatedVertices(AnimationMode_Deform, d_temp_vertices);
    //
    //    // Update deforming GAS
    //
    //    OptixAccelBuildOptions gas_accel_options = {};
    //    gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    //    gas_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    ////
    //    //std::cout<<"\n\n\n\r***************************************\n";
    //    //std::cout<<"size out buffer0: "<<instances[0].gas_output_buffer_size<<std::endl;
    //    //std::cout<<"size out buffer1: "<<instances[1].gas_output_buffer_size<<std::endl;
    //    OptixTraversableHandle deforming_gas_handle;
    //    OPTIX_CHECK( optixAccelBuild(
    //             context,
    //             stream,                       // CUDA stream
    //            &gas_accel_options,
    //            & instances[1].build_input,
    //            1,                                  // num build inputs
    //             d_temp_buffer,
    //             temp_buffer_size,
    //            instances[1].d_gas_output_buffer,
    //            instances[1].gas_output_buffer_size,
    //            & deforming_gas_handle,
    //            nullptr,                           // emitted property list
    //            0                                   // num emitted properties
    //            ) );
    //    instances[1].gas_handle =deforming_gas_handle;
    //
    //    // Generate exploding sphere vertices
    //    launchGenerateAnimatedVertices(AnimationMode_Explode, d_temp_vertices);
    //
    //    // Update exploding GAS
    //
    //    // Occasionally rebuild to maintain AS quality
    //if(  time -  last_exploding_sphere_rebuild_time > 1 / g_exploding_gas_rebuild_frequency )
    //{
    //    gas_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    //     last_exploding_sphere_rebuild_time =  time;
    //
    //    // We don't compress the AS so the size of the GAS won't change and we can rebuild the GAS in-place.
    //}
    //
    //
    //    std::cout<<instances.size()<<std::endl;
    //    OPTIX_CHECK( optixAccelBuild(
    //             context,
    //             stream,                       // CUDA stream
    //            &gas_accel_options,
    //            & instances[0].build_input,
    //            1,                                  // num build inputs
    //             d_temp_buffer,
    //             temp_buffer_size,
    //             instances[0].d_gas_output_buffer,
    //             instances[0].gas_output_buffer_size,
    //            & instances[0].gas_handle,
    //            nullptr,                           // emitted property list
    //            0                                   // num emitted properties
    //            ) );
*/
    if (g_instances.size() > 1) {
        float t = sinf(time * 4.f);
        float t1 = cosf(time * 4.f);
        for (int i = 0; i < 0; i++) {
            float transform[12] = {
                    // Row 0
                    1, 0, 0, 0,// R₀₀, R₀₁, R₀₂, T₀
                    // Row 1
                    0, 1, 0, 0,// R₁₀, R₁₁, R₁₂, T₁
                    // Row 2
                    0, 0, 1, 0// R₂₀, R₂₁, R₂₂, T₂
            };
            setRotation(time * 4.f, time * 4.f, 0.f, transform);
            // Optionally set translation components
            transform[3] = (i - 5) * 1.5;// T₀ (translation along X-axis)
            transform[7] = t1;           // T₁ (translation along Y-axis)
            transform[11] = 0.0f;        // T₂ (translation along Z-axis)
            CUDA_CHECK(cudaMemcpy(((OptixInstance *) d_instances)[i].transform, transform, sizeof(float) * 12, cudaMemcpyHostToDevice));
        }
    }

    // Update the IAS
    // We refit the IAS as the relative positions of the spheres don't change much so AS quality after update is fine.

    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    ias_accel_options.motionOptions.numKeys = 1;
    ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OPTIX_CHECK(optixAccelBuild(context, stream, &ias_accel_options, &ias_instance_input, 1, d_temp_buffer, temp_buffer_size,
                                d_ias_output_buffer, ias_output_buffer_size, &ias_handle, nullptr, 0));

    CUDA_SYNC_CHECK();
}


int num_frames = 16;
float animation_time = 1.f;
int i = 0;
void OptixRenderer::render(torch::Tensor &output_tensor, const std::vector<float> &camera_params) {
    updateSBT();
    if (camera_params.size() != 9) {
        throw std::runtime_error("camera_params must have 9 elements: eye(3), lookat(3), up(3)");
    }

    float3 eye = make_float3(camera_params[0], camera_params[1], camera_params[2]);
    float3 lookat = make_float3(camera_params[3], camera_params[4], camera_params[5]);
    float3 up = make_float3(camera_params[6], camera_params[7], camera_params[8]);

    // Set up camera
    float fov = 90.0f;
    float aspect_ratio = static_cast<float>(params.width) / static_cast<float>(params.height);
    float3 W = normalize(eye - lookat);
    float3 U = normalize(cross(up, W));
    float3 V = cross(W, U);

    float ulen = tanf(fov * 0.5f * static_cast<float>(M_PI) / 180.0f);
    U *= ulen;
    V *= ulen / aspect_ratio;

    params.eye = eye;//make_float3(0.f, 1.f, -20.f);//eye;
    params.U   = U  ;  //make_float3(-8.41847f, 0.f, 0.f);
    params.V   = V  ;  //make_float3(0.f, 6.30598f, 0.315299f);
    params.W   = W  ;  //make_float3(0.f, -1.f, 20.f);

    time = i * (animation_time / (num_frames - 1)) / 5;

    updateMeshAccel();
    //updateState(output_tensor);
    launchSubframe(output_tensor);
    ++params.subframe_index;
    i++;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Geometry>(m, "Geometry")
            .def("setMaterialColor", &Geometry::setMaterialColor)
            .def("setTexture", &Geometry::setTexture);
    py::class_<TextureObject>(m, "TextureObject")
            .def(py::init<torch::Tensor&>());

    py::class_<OptixRenderer>(m, "Renderer")
            .def(py::init<int, int>())
            .def("render", &OptixRenderer::render)
            .def("createVertexGeometry",
                 [](OptixRenderer& self, torch::Tensor& vertices,
                                            py::object indices,
                                            py::object texCoords,
                                            py::object tangents,
                                            py::object bitangents,
                                            py::object vertex_normals,
                                            py::object textureObject,
                                            py::object normals,
                                            py::object metallic_roughness,
                                            py::object emission_texture){
                     torch::Tensor* texCoordsPointer =texCoords.is_none() ? nullptr : &texCoords.cast<torch::Tensor>();
                     torch::Tensor* indicesPointer = indices.is_none() ? nullptr : &indices.cast<torch::Tensor>();
                     torch::Tensor* tangentsPointer = tangents.is_none() ? nullptr : &tangents.cast<torch::Tensor>();
                     torch::Tensor* bitangentsPointer = bitangents.is_none() ? nullptr : &bitangents.cast<torch::Tensor>();
                     torch::Tensor* vertex_normalsPointer = vertex_normals.is_none() ? nullptr : &vertex_normals.cast<torch::Tensor>();
                     TextureObject* normalsObjectPointer = normals.is_none() ? nullptr : &normals.cast<TextureObject>();
                     TextureObject* textureObjectPointer = textureObject.is_none() ? nullptr : &textureObject.cast<TextureObject>();
                     TextureObject* metallic_roughnessObjectPointer = metallic_roughness.is_none() ? nullptr : &metallic_roughness.cast<TextureObject>();
                     TextureObject* emission_textureObjectPointer = emission_texture.is_none() ? nullptr : &emission_texture.cast<TextureObject>();

                    std::cout<<"geometry ptrs: "<<texCoordsPointer<<" "<<indicesPointer<<" "<<textureObjectPointer<<std::endl;
                     // Call the actual createVertexGeometry method with the appropriate arguments
                     return self.createVertexGeometry(vertices, indicesPointer, texCoordsPointer, tangentsPointer, bitangentsPointer,
                                                     vertex_normalsPointer, textureObjectPointer, normalsObjectPointer,
                                                     metallic_roughnessObjectPointer, emission_textureObjectPointer);
                 },
                 py::arg("vertices"), py::arg("indices") = py::none(), py::arg("texCoords") = py::none(),
                 py::arg("tangents") = py::none(), py::arg("bitangents") = py::none(), py::arg("vertex_normals") = py::none(), py::arg("textureObject") = py::none(),
                 py::arg("normals") = py::none(), py::arg("metallic_roughness") = py::none(), py::arg("emission_texture") = py::none(),
                 "Create a vertex geometry with optional texture coordinates and texture object.")
            .def("addGeometryInstance", &OptixRenderer::addGeometryInstance)
            .def("buidIAS", &OptixRenderer::buidIAS)
            .def("getTransformForInstance", &OptixRenderer::getTransformForInstance);


    //        .def(py::init<torch::Tensor, torch::Tensor>())
    //        .def("render", &OptixRenderer::render)
    //        .def("set_geometry", &OptixRenderer::set_geometry);
}
