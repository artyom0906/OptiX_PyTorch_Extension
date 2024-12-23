// renderer.cpp

#include "renderer.h"

// Structures for SBT records
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RayGenRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Additional data if needed
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Additional data if needed
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitGroupRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // Additional data if needed
};

// Constructor
Renderer::Renderer(int width, int height){
    initOptix();
    createContext();
    buildModule();
    createProgramGroups();
    createPipeline();
    buildSBT();

    // Allocate params buffer
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
}

// Destructor
Renderer::~Renderer() {
    // Free device memory
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_raygen_record)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_miss_record)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hitgroup_record)));
    // Do not free d_vertices and d_indices; they are managed by PyTorch
    if (d_gas_output_buffer) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
    }
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));

    // Destroy OptiX objects
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));
}

// Initialization methods
void Renderer::initOptix() {
    // Initialize OptiX
    OPTIX_CHECK(optixInit());
}
#include <iostream>
#include <cstdarg> // For va_list and related functions

// OptiX log callback function
void optix_log_callback(unsigned int level, const char* tag, const char* message, void* cbdata) {
    // Print the log message to standard output
    std::cerr << "[OptiX][Level " << level << "][" << tag << "] " << message << std::endl;
}

void Renderer::createContext() {
    // Create OptiX device context
    CUcontext cuCtx = 0;  // Zero means take the current context
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optix_log_callback;
    options.logCallbackData = nullptr;
    options.logCallbackLevel = 4;  // Default log level
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
}

void Renderer::buildModule() {
    // Build the module from PTX
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    // Load PTX code from file
    const char* ptx_filename = "optix_programs.ptx";
    std::ifstream ptx_file(ptx_filename);
    if (!ptx_file.is_open()) {
        std::cerr << "Failed to open PTX file: " << ptx_filename << std::endl;
        exit(1);
    }
    std::string ptx_source((std::istreambuf_iterator<char>(ptx_file)), std::istreambuf_iterator<char>());
    ptx_file.close();

    // Create module from PTX
    OPTIX_CHECK_LOG(optixModuleCreate(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            ptx_source.c_str(),
            ptx_source.size(),
            LOG,
            &LOG_SIZE,
            &module
            ));
}

void Renderer::createProgramGroups() {
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupOptions program_group_options = {};

    // Raygen program group
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,
            &program_group_options,
            LOG,
            &LOG_SIZE,
            &raygen_prog_group
            ));

    // Miss program group
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,
            &program_group_options,
            LOG,
            &LOG_SIZE,
            &miss_prog_group
            ));

    // Hitgroup program group
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,
            &program_group_options,
            LOG,
            &LOG_SIZE,
            &hitgroup_prog_group
            ));
}

void Renderer::createPipeline() {
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;

    OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

    OPTIX_CHECK_LOG(optixPipelineCreate(
            context,
            &pipeline_compile_options,  // pipeline compile options already specified
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups)/sizeof(program_groups[0]),
            LOG,
            &LOG_SIZE,
            &pipeline
            ));

    // Set stack size
    OPTIX_CHECK(optixPipelineSetStackSize(
            pipeline,
            2*1024,   // direct stack size from RG
            2*1024,   // direct stack size from CH and MS
            2*1024,   // continuation stack size
            1         // maxTraversableDepth
            ));
}

void Renderer::buildAccelerationStructure() {
    // Acceleration structure build options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_REFIT;
    accel_options.operation = gas_handle ? OPTIX_BUILD_OPERATION_REFIT : OPTIX_BUILD_OPERATION_BUILD;

    // Geometry input
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &d_vertices;
    build_input.triangleArray.numVertices = num_vertices;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.indexBuffer = d_indices;
    build_input.triangleArray.numIndexTriplets = num_indices / 3;
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);

    // Specify geometry flags
    std::vector<uint32_t> triangle_flags(1, OPTIX_GEOMETRY_FLAG_NONE);
    build_input.triangleArray.flags = triangle_flags.data();
    build_input.triangleArray.numSbtRecords = 1;

    // Compute buffer sizes
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &buffer_sizes));

    // Allocate temporary buffer
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), buffer_sizes.tempSizeInBytes));

    // Allocate output buffer only if needed
    if (!d_gas_output_buffer) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), buffer_sizes.outputSizeInBytes));
    }

    // Build or refit GAS
    OPTIX_CHECK(optixAccelBuild(
            context,
            0, // CUDA stream
            &accel_options,
            &build_input,
            1, // Number of build inputs
            d_temp_buffer,
            buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer,
            buffer_sizes.outputSizeInBytes,
            &gas_handle,
            nullptr, // Compacted size output buffer
            0 // Compacted size output buffer size
            ));

    // Wait for the build to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary buffers
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
}

void Renderer::buildSBT() {
    // Create SBT records
    // Raygen record
    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RayGenRecord)));
    CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_raygen_record),
            &rg_sbt,
            sizeof(RayGenRecord),
            cudaMemcpyHostToDevice
            ));

    // Miss record
    MissRecord ms_sbt = {

    };
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_miss_record),
            &ms_sbt,
            sizeof(MissRecord),
            cudaMemcpyHostToDevice
            ));

    // Hitgroup record
    HitGroupRecord hg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), sizeof(HitGroupRecord)));
    CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_hitgroup_record),
            &hg_sbt,
            sizeof(HitGroupRecord),
            cudaMemcpyHostToDevice
            ));

    // Set up the SBT
    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = d_hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt.hitgroupRecordCount = 1;
}

void Renderer::set_geometry(const torch::Tensor& vertices, const torch::Tensor& indices) {
    // Ensure tensors are on GPU and correct type
    TORCH_CHECK(vertices.is_cuda(), "Vertices tensor must be a CUDA tensor.");
    TORCH_CHECK(indices.is_cuda(), "Indices tensor must be a CUDA tensor.");
    TORCH_CHECK(vertices.scalar_type() == torch::kFloat32, "Vertices tensor must be of type float32.");
    TORCH_CHECK(indices.scalar_type() == torch::kUInt32, "Indices tensor must be of type uint32.");

    // Update device pointers
    d_vertices = reinterpret_cast<CUdeviceptr>(vertices.data_ptr<float>());
    d_indices = reinterpret_cast<CUdeviceptr>(indices.data_ptr());

    // Update geometry info
    num_vertices = vertices.size(0);
    num_indices = indices.size(0);

    // Do not rebuild AS here; we'll build it externally
}

// Render method
void Renderer::render(torch::Tensor& output_tensor, const std::vector<float>& camera_params) {
    CUDA_CHECK(cudaDeviceSynchronize());
    // Extract camera parameters
    if (camera_params.size() != 9) {
        throw std::runtime_error("camera_params must have 9 elements: eye(3), lookat(3), up(3)");
    }

    float3 eye = make_float3(camera_params[0], camera_params[1], camera_params[2]);
    float3 lookat = make_float3(camera_params[3], camera_params[4], camera_params[5]);
    float3 up = make_float3(camera_params[6], camera_params[7], camera_params[8]);

    // Set up camera
    float fov = 90.0f;
    float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    float3 W = normalize(eye - lookat);
    float3 U = normalize(cross(up, W));
    float3 V = cross(W, U);

    float ulen = tanf(fov * 0.5f * static_cast<float>(M_PI) / 180.0f);
    U *= ulen;
    V *= ulen / aspect_ratio;

    // Set up launch parameters
    Params params;
    params.width = width;
    params.height = height;
    params.image = reinterpret_cast<float3*>(output_tensor.data_ptr<float>());
    params.eye = eye;
    params.U = U;
    params.V = V;
    params.W = W;
    params.handle = gas_handle;

    // Copy params to device
    CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_params),
            &params,
            sizeof(Params),
            cudaMemcpyHostToDevice
            ));

    CUDA_CHECK(cudaDeviceSynchronize());
    // Launch OptiX kernel
    OPTIX_CHECK(optixLaunch(
            pipeline,
            0, // CUDA stream
            d_params,
            sizeof(Params),
            &sbt,
            width,
            height,
            1 // depth
            ));

    // Wait for CUDA to finish
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Expose the Renderer class to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Renderer>(m, "Renderer")
            .def(py::init<int, int>())
            .def("render", &Renderer::render)
            .def("set_geometry", &Renderer::set_geometry)
            .def("buildSBT", &Renderer::buildSBT)
            .def("buildAccelerationStructure", &Renderer::buildAccelerationStructure);
}
