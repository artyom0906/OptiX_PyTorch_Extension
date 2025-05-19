//
// Created by artem on 4/7/25.
//

#include "../include/OpenVRSystem.cuh"

// Global variables to store timing data between functions
float leftEyeRenderTime = 0.0f;
float leftEyeInternalCopyTime = 0.0f; 
float leftEyeTextureCopyTime = 0.0f;
float leftEyeToCpuCopyTime = 0.0f;    // Time to copy from GPU to CPU
float leftEyeFromCpuCopyTime = 0.0f;  // Time to copy from CPU to GPU

float rightEyeRenderTime = 0.0f;
float rightEyeInternalCopyTime = 0.0f;
float rightEyeTextureCopyTime = 0.0f;
float rightEyeToCpuCopyTime = 0.0f;   // Time to copy from GPU to CPU
float rightEyeFromCpuCopyTime = 0.0f; // Time to copy from CPU to GPU

// Total CPU copy times for the frame
float totalToCpuCopyTime = 0.0f;
float totalFromCpuCopyTime = 0.0f;

// Initialize performance logging to CSV file
bool OpenVRSystem::InitPerfLogging() {
    // Open the file for writing
    m_perfLogFile.open(m_perfLogFilename, std::ios::out);
    
    if (!m_perfLogFile.is_open()) {
        std::cerr << "Failed to open performance log file: " << m_perfLogFilename << std::endl;
        return false;
    }
    
    // CSV header row - column names depend on whether units are included in values
#ifdef CSV_INCLUDE_UNITS
    m_perfLogFile << "Frame,Time,LeftEyeRenderTime,LeftEyeInternalCopyTime,LeftEyeTextureCopyTime,"
                  << "LeftEyeToCpuCopyTime,LeftEyeFromCpuCopyTime,"
                  << "RightEyeRenderTime,RightEyeInternalCopyTime,RightEyeTextureCopyTime,"
                  << "RightEyeToCpuCopyTime,RightEyeFromCpuCopyTime,"
                  << "TotalToCpuCopyTime,TotalFromCpuCopyTime,"
                  << "TotalFrameTime,FPS" << std::endl;
#else
    m_perfLogFile << "Frame,Time,LeftEyeRenderTime_ms,LeftEyeInternalCopyTime_ms,LeftEyeTextureCopyTime_ms,"
                  << "LeftEyeToCpuCopyTime_ms,LeftEyeFromCpuCopyTime_ms,"
                  << "RightEyeRenderTime_ms,RightEyeInternalCopyTime_ms,RightEyeTextureCopyTime_ms,"
                  << "RightEyeToCpuCopyTime_ms,RightEyeFromCpuCopyTime_ms,"
                  << "TotalToCpuCopyTime_ms,TotalFromCpuCopyTime_ms,"
                  << "TotalFrameTime_ms,FPS" << std::endl;
#endif
    
    // Always show this message even in non-debug mode since it's important for the user to know
    std::cout << "VR performance logging enabled: " << m_perfLogFilename << std::endl;
#ifdef CSV_INCLUDE_UNITS
    std::cout << "CSV format: Units included with each value (e.g., '10.123 ms')" << std::endl;
#else
    std::cout << "CSV format: Units in column headers only (e.g., 'RenderTime_ms')" << std::endl;
#endif
    return true;
}

// CUDA kernel to convert RGB float tensor data to RGBA bytes in a CUDA surface
__global__ void convertTensorToSurface(float* tensorData, 
                                      cudaSurfaceObject_t surface,
                                      int width, 
                                      int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Get RGB values from tensor data (layout: HWC)
    // Each pixel has 3 float values (RGB)
    int tensor_idx = (y * width + x) * 3;
    float r = tensorData[tensor_idx];
    float g = tensorData[tensor_idx + 1];
    float b = tensorData[tensor_idx + 2];
    
    // Convert 0-1 float values to 0-255 bytes
    unsigned char r_byte = static_cast<unsigned char>(r * 255.0f);
    unsigned char g_byte = static_cast<unsigned char>(g * 255.0f);
    unsigned char b_byte = static_cast<unsigned char>(b * 255.0f);
    unsigned char a_byte = 255; // Always fully opaque
    
    // Pack RGBA bytes into a single uint32
    uchar4 rgba;
    rgba.x = b_byte; // B (BGRA format for OpenGL texture)
    rgba.y = g_byte; // G
    rgba.z = r_byte; // R
    rgba.w = a_byte; // A
    
    // Write to surface (y-coordinate is flipped for OpenGL convention)
    surf2Dwrite(rgba, surface, x * sizeof(uchar4), height - 1 - y);
}

// CUDA kernel for rendering a grid pattern to the left eye (red and black)
__global__ void renderLeftEyeGrid(unsigned char* surface,
                                  int            width,
                                  int            height,
                                  int            pitch,
                                  float          time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Calculate output position in the surface
    unsigned char* pixel = surface + y * pitch + x * 4;

    // Create a grid pattern that moves slowly over time
    int gridSize = 32;// Size of each grid cell
    int offsetX  = (int)(time * 10.0f) % gridSize;
    int offsetY  = (int)(time * 5.0f) % gridSize;

    bool isGridLine = ((x + offsetX) % gridSize < 2) || ((y + offsetY) % gridSize < 2);

    if (isGridLine) {
        // Grid lines: Red
        pixel[0] = 0;  // B
        pixel[1] = 0;  // G
        pixel[2] = 255;// R
        pixel[3] = 255;// A
    } else {
        // Grid cells: Black
        pixel[0] = 0;  // B
        pixel[1] = 0;  // G
        pixel[2] = 0;  // R
        pixel[3] = 255;// A
    }
}

// CUDA kernel for rendering a grid pattern to the right eye (blue and black)
__global__ void renderRightEyeGrid(unsigned char* surface,
                                   int            width,
                                   int            height,
                                   int            pitch,
                                   float          time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Calculate output position in the surface
    unsigned char* pixel = surface + y * pitch + x * 4;

    // Create a grid pattern that moves slowly over time (different direction from left eye)
    int gridSize = 32;// Size of each grid cell
    int offsetX  = (int)(time * 5.0f) % gridSize;
    int offsetY  = (int)(time * 10.0f) % gridSize;

    bool isGridLine = ((x + offsetX) % gridSize < 2) || ((y + offsetY) % gridSize < 2);

    if (isGridLine) {
        // Grid lines: Blue
        pixel[0] = 255;// B
        pixel[1] = 0;  // G
        pixel[2] = 0;  // R
        pixel[3] = 255;// A
    } else {
        // Grid cells: Black
        pixel[0] = 0;  // B
        pixel[1] = 0;  // G
        pixel[2] = 0;  // R
        pixel[3] = 255;// A
    }
}

// CUDA error checking helper
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

bool OpenVRSystem::Initialize() {
    // Initialize GLFW and create a window
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    m_window = glfwCreateWindow(800, 600, "OpenVR CUDA Grid Example", nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum glewError = glewInit();
    if (glewError != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(glewError) << std::endl;
        return false;
    }

    // Initialize OpenVR
    vr::EVRInitError eError = vr::VRInitError_None;
    m_pVRSystem             = vr::VR_Init(&eError, vr::VRApplication_Scene);
    if (eError != vr::VRInitError_None) {
        std::cerr << "Failed to initialize OpenVR: "
            << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;
        return false;
    }

    DEBUG_PRINT("OpenVR initialized successfully");

    // Get the recommended render target size
    m_pVRSystem->GetRecommendedRenderTargetSize(&m_renderWidth, &m_renderHeight);
    DEBUG_PRINT("Recommended render target size: " << m_renderWidth << "x" << m_renderHeight);

    // Initialize VR compositor
    if (!vr::VRCompositor()) {
        std::cerr << "Failed to initialize VR compositor" << std::endl;
        vr::VR_Shutdown();
        return false;
    }

    // Create OpenGL textures and register them with CUDA
    if (!CreateTextureForEye(m_leftEye) || !CreateTextureForEye(m_rightEye)) {
        std::cerr << "Failed to create textures" << std::endl;
        return false;
    }
    
    // Initialize performance logging
    m_frameCount = 0;
    if (!InitPerfLogging()) {
        std::cerr << "Warning: Failed to initialize performance logging. Continuing without logging." << std::endl;
    }

    return true;
}

void OpenVRSystem::Shutdown() {
    // Close performance log file if open
    if (m_perfLogFile.is_open()) {
        m_perfLogFile.close();
        // Always show this message even in non-debug mode
        std::cout << "VR performance log complete: " << m_perfLogFilename 
                  << " (" << m_frameCount << " frames)" << std::endl;
    }
    
    // Unregister CUDA resources
    if (m_leftEye.cudaResource) {
        cudaGraphicsUnregisterResource(m_leftEye.cudaResource);
        m_leftEye.cudaResource = nullptr;
    }

    if (m_rightEye.cudaResource) {
        cudaGraphicsUnregisterResource(m_rightEye.cudaResource);
        m_rightEye.cudaResource = nullptr;
    }

    // Delete OpenGL textures
    if (m_leftEye.texture) {
        glDeleteTextures(1, &m_leftEye.texture);
        m_leftEye.texture = 0;
    }

    if (m_rightEye.texture) {
        glDeleteTextures(1, &m_rightEye.texture);
        m_rightEye.texture = 0;
    }

    // Shutdown OpenVR
    if (m_pVRSystem) {
        vr::VR_Shutdown();
        m_pVRSystem = nullptr;
    }

    // Destroy GLFW window and terminate GLFW
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
        glfwTerminate();
    }
}

bool OpenVRSystem::CreateTextureForEye(EyeResources& eye) const {
    // Create OpenGL texture
    glGenTextures(1, &eye.texture);
    glBindTexture(GL_TEXTURE_2D, eye.texture);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Allocate storage for the texture
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 m_renderWidth,
                 m_renderHeight,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 nullptr);

    // Make sure OpenGL commands are completed
    glFinish();
    
    // Use the device context if available
    bool useDeviceContext = m_deviceContext && m_deviceContext->isInitialized();
    
    if (useDeviceContext) {
        // Set the device context to ensure proper CUDA context
        if (!m_deviceContext->setDevice()) {
            std::cerr << "Failed to set device context for texture registration" << std::endl;
            useDeviceContext = false;
        } else {
            DEBUG_PRINT("Using device context for device " << m_deviceId << " to register texture");
        }
    }
    
    if (!useDeviceContext) {
        // Fallback to basic CUDA device selection
        cudaSetDevice(m_deviceId);
        DEBUG_PRINT("Fallback: Using cudaSetDevice(" << m_deviceId << ") for texture registration");
    }

    // Register the texture with CUDA
    const cudaError_t cudaStatus = cudaGraphicsGLRegisterImage(
        &eye.cudaResource,
        eye.texture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard
        );

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to register texture with CUDA: "
            << cudaGetErrorString(cudaStatus) << std::endl;
        if (useDeviceContext) {
            m_deviceContext->releaseDevice();
        }
        return false;
    }

    DEBUG_PRINT("Successfully registered texture " << eye.texture << " with CUDA resource: " << eye.cudaResource);
    
    // Release the device context if we used it
    if (useDeviceContext) {
        m_deviceContext->releaseDevice();
    }
    
    return true;
}

void OpenVRSystem::RenderFrame() {
    // Use the device context if available, otherwise set device directly
    bool useDeviceContext = m_deviceContext && m_deviceContext->isInitialized();
    CUstream stream = 0; // Default stream
    
    if (useDeviceContext) {
        if (!m_deviceContext->setDevice()) {
            std::cerr << "Failed to set device context for frame timing" << std::endl;
            useDeviceContext = false;
            CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
        } else {
            stream = m_deviceContext->getStream();
        }
    } else {
        CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
    }
    
    // Create CUDA events for frame timing in the correct context
    cudaEvent_t startFrame, endFrame;
    CHECK_CUDA_ERROR(cudaEventCreate(&startFrame));
    CHECK_CUDA_ERROR(cudaEventCreate(&endFrame));
    
    // Reset CPU copy timing totals at the start of each frame
    totalToCpuCopyTime = 0.0f;
    totalFromCpuCopyTime = 0.0f;
    
    // Start timing the full frame with proper stream
    CHECK_CUDA_ERROR(cudaEventRecord(startFrame, stream));
    
    // Update time
    m_time += 0.01f;
    
    // Process VR events
    vr::VREvent_t event{};
    while (m_pVRSystem->PollNextEvent(&event, sizeof(event))) {
        ProcessVREvent(event);
    }
    
    // Wait for poses to synchronize with the display
    vr::TrackedDevicePose_t trackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
    vr::TrackedDevicePose_t gamePoses[vr::k_unMaxTrackedDeviceCount];
    vr::VRCompositor()->WaitGetPoses(trackedDevicePoses,
                                     vr::k_unMaxTrackedDeviceCount,
                                     gamePoses,
                                     vr::k_unMaxTrackedDeviceCount);
    
    // Log head position from the HMD (device 0)
    if (trackedDevicePoses[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
        const auto& pose = trackedDevicePoses[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking;
        
        // Extract position from the pose matrix
        float posX = pose.m[0][3];
        float posY = pose.m[1][3];
        float posZ = pose.m[2][3];
        
        // Log HMD position every 30 frames
        static int frame_count = 0;
        if (frame_count++ % 30 == 0) {
            DEBUG_PRINT("HMD Position: " << posX << ", " << posY << ", " << posZ);
            
            // Also log rotation matrix for debugging
            DEBUG_PRINT("HMD Rotation: [" 
                      << pose.m[0][0] << ", " << pose.m[0][1] << ", " << pose.m[0][2] << "]");
            DEBUG_PRINT("              [" 
                      << pose.m[1][0] << ", " << pose.m[1][1] << ", " << pose.m[1][2] << "]");
            DEBUG_PRINT("              [" 
                      << pose.m[2][0] << ", " << pose.m[2][1] << ", " << pose.m[2][2] << "]");
        }
    } else {
        DEBUG_PRINT("HMD pose not valid");
    }

        // Render left eye
        RenderEyeTexture(vr::Eye_Left);

        // Render right eye
        RenderEyeTexture(vr::Eye_Right);

        // Submit to compositor
        SubmitFramesToCompositor();
        
        // Make sure we're still in the same CUDA context before ending the timing
        if (useDeviceContext) {
            if (!m_deviceContext->setDevice()) {
                std::cerr << "Failed to set device context for ending frame timing" << std::endl;
                useDeviceContext = false;
                CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
                stream = 0; // Use default stream as fallback
            }
        } else {
            CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
        }
        
        // End timing for full frame with proper stream
        CHECK_CUDA_ERROR(cudaEventRecord(endFrame, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(endFrame));
        
        // Calculate and log full frame time
        float frameTimeMs = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&frameTimeMs, startFrame, endFrame));
        
        // Calculate frame rate
        float frameRate = 1000.0f / frameTimeMs;
        
        // Log the total frame time periodically to console
        static int timing_frame_count = 0;
        if (timing_frame_count++ % 10 == 0) {
            DEBUG_PRINT("===== TOTAL VR FRAME TIME: " << std::fixed << std::setprecision(3) << frameTimeMs << " ms =====");
            DEBUG_PRINT("===== FRAME RATE: " << std::fixed << std::setprecision(2) << frameRate << " FPS =====");
            
            // Log CPU copy times as well
            if (totalToCpuCopyTime > 0.0f || totalFromCpuCopyTime > 0.0f) {
                DEBUG_PRINT("===== CPU COPY TIMES: To CPU: " << std::fixed << std::setprecision(3) 
                            << totalToCpuCopyTime << " ms, From CPU: " 
                            << totalFromCpuCopyTime << " ms =====");
            }
        }
        
        // No need for extern declarations since we're using global variables
        
        // Write performance data to CSV file if open
        if (m_perfLogFile.is_open()) {
            // Current time with millisecond precision
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch() % std::chrono::seconds(1));
            
            // Format time with milliseconds
            char timestamp[32];
            std::tm timeinfo;
            localtime_r(&time_t_now, &timeinfo);  // Thread-safe version
            std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", &timeinfo);
            
            // Create timestamp with millisecond precision
            char precise_timestamp[40];
            snprintf(precise_timestamp, sizeof(precise_timestamp), "%s.%03d", 
                     timestamp, static_cast<int>(ms.count()));
            
            // Set fixed precision for floating point values (3 decimal places)
            m_perfLogFile << std::fixed << std::setprecision(3);
            
#ifdef CSV_INCLUDE_UNITS
            // Write CSV row with millisecond precision and "ms" units after each timing value
            m_perfLogFile << m_frameCount << ","
                          << precise_timestamp << ","
                          << leftEyeRenderTime << " ms," 
                          << leftEyeInternalCopyTime << " ms,"
                          << leftEyeTextureCopyTime << " ms,"
                          << leftEyeToCpuCopyTime << " ms,"
                          << leftEyeFromCpuCopyTime << " ms,"
                          << rightEyeRenderTime << " ms,"
                          << rightEyeInternalCopyTime << " ms,"
                          << rightEyeTextureCopyTime << " ms,"
                          << rightEyeToCpuCopyTime << " ms,"
                          << rightEyeFromCpuCopyTime << " ms,"
                          << totalToCpuCopyTime << " ms,"
                          << totalFromCpuCopyTime << " ms,"
                          << frameTimeMs << " ms,"
                          << frameRate << std::endl;
#else
            // Write CSV row with millisecond precision but no units (for easier data processing)
            m_perfLogFile << m_frameCount << ","
                          << precise_timestamp << ","
                          << leftEyeRenderTime << ","
                          << leftEyeInternalCopyTime << ","
                          << leftEyeTextureCopyTime << ","
                          << leftEyeToCpuCopyTime << ","
                          << leftEyeFromCpuCopyTime << ","
                          << rightEyeRenderTime << ","
                          << rightEyeInternalCopyTime << ","
                          << rightEyeTextureCopyTime << ","
                          << rightEyeToCpuCopyTime << ","
                          << rightEyeFromCpuCopyTime << ","
                          << totalToCpuCopyTime << ","
                          << totalFromCpuCopyTime << ","
                          << frameTimeMs << ","
                          << frameRate << std::endl;
#endif
            
            // Increment frame counter
            m_frameCount++;
        }
        
        // Clean up CUDA events
        CHECK_CUDA_ERROR(cudaEventDestroy(startFrame));
        CHECK_CUDA_ERROR(cudaEventDestroy(endFrame));
}

void OpenVRSystem::RenderEyeTexture(vr::EVREye eye) {
    EyeResources& eyeRes = (eye == vr::Eye_Left) ? m_leftEye : m_rightEye;
    optix_renderer::Renderer* renderer = (eye == vr::Eye_Left) ?
                                          m_leftEyeRenderer : m_rightEyeRenderer;
    const char* eyeName = (eye == vr::Eye_Left) ? "Left" : "Right";
    
    // We're using global variables for timing data now
    
    // Create variables for timing
    cudaEvent_t startRender = 0, endRender = 0, startCopy = 0, endCopy = 0;
    float renderTime = 0.0f, copyTime = 0.0f, textureCopyTime = 0.0f;
    if (renderer) {
        // Use the device context from the resource manager
        bool useDeviceContext = m_deviceContext && m_deviceContext->isInitialized();
        CUstream stream = 0; // Default stream
        
        if (useDeviceContext) {
            // Set the device context to ensure proper CUDA context
            if (!m_deviceContext->setDevice()) {
                std::cerr << "Failed to set device context for rendering" << std::endl;
                useDeviceContext = false;
            } else {
                //std::cout << "Using device context for rendering" << std::endl;
                stream = m_deviceContext->getStream();
            }
        } else {
            // Fallback to setting the device directly
            CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
        }
        
        // Create timing events in the correct CUDA context
        CHECK_CUDA_ERROR(cudaEventCreate(&startRender));
        CHECK_CUDA_ERROR(cudaEventCreate(&endRender));
        CHECK_CUDA_ERROR(cudaEventCreate(&startCopy));
        CHECK_CUDA_ERROR(cudaEventCreate(&endCopy));
        
        // Render with OptiX, always rebuilding the acceleration structure to ensure transforms are updated
        torch::Tensor result = renderer->render(m_renderWidth, m_renderHeight, true);
        
        // Get timing information from the renderer
        float renderTimeMs = renderer->getLastRenderTimeMs();
        float copyTimeMs = renderer->getLastCopyTimeMs();
        
        // Save timing data for CSV logging
        if (eye == vr::Eye_Left) {
            leftEyeRenderTime = renderTimeMs;
            leftEyeInternalCopyTime = copyTimeMs;
        } else {
            rightEyeRenderTime = renderTimeMs;
            rightEyeInternalCopyTime = copyTimeMs;
        }

        // --- Populate CPU tensor for Python monitor ---
        if (eye == vr::Eye_Left) {
            m_lastLeftEyeTensor_CPU = result.cpu().clone();
        } else {
            m_lastRightEyeTensor_CPU = result.cpu().clone();
        }
        
        // Print to console only in debug mode with fixed precision (3 decimal places)
        DEBUG_PRINT(eyeName << " Eye - Render time: " << std::fixed << std::setprecision(3) << renderTimeMs 
                  << " ms, Internal copy time: " << std::fixed << std::setprecision(3) << copyTimeMs << " ms");
        
        // Make sure we're still in the right CUDA context before recording the event
        if (useDeviceContext) {
            if (!m_deviceContext->setDevice()) {
                std::cerr << "Failed to set device context for texture copy timing" << std::endl;
                useDeviceContext = false;
                CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
                stream = 0; // Use default stream as fallback
            }
        } else {
            CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
        }
        
        // Start timing the GPU buffer mapping and transfer operations with proper stream
        CHECK_CUDA_ERROR(cudaEventRecord(startCopy, stream));
        
        // Ensure we're in the right CUDA context for resource mapping
        if (useDeviceContext) {
            if (!m_deviceContext->setDevice()) {
                std::cerr << "Failed to set device context for resource mapping" << std::endl;
                useDeviceContext = false;
                cudaSetDevice(m_deviceId);
            }
        } else {
            // Fallback to setting the device directly
            CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
        }
        
        // Map the texture for CUDA access
        cudaArray_t cudaArray;
        //std::cout << "Mapping CUDA graphics resource with "
        //          << (useDeviceContext ? "device context stream" : "default stream") << std::endl;
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &eyeRes.cudaResource, stream));
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaArray,
            eyeRes.cudaResource, 0, 0));

        // Get tensor data pointer
        void* tensor_ptr = nullptr;
        size_t buffer_size = m_renderWidth * m_renderHeight * 3 * sizeof(float);

        // Move tensor to the current GPU if needed
        torch::Tensor gpu_result;
        if (result.is_cuda() && result.device().index() != m_deviceId) {
            //std::cout << "Moving tensor from device " << result.device().index()
            //        << " to device " << m_deviceId << std::endl;
            void* source_ptr = result.data_ptr();
            if (!renderer->getDeviceContext()->setDevice()) {
                exit(-1);
            }
            
            // Create and start timing for GPU to CPU copy
            cudaEvent_t startToCpu, stopToCpu;
            CHECK_CUDA_ERROR(cudaEventCreate(&startToCpu));
            CHECK_CUDA_ERROR(cudaEventCreate(&stopToCpu));
            CHECK_CUDA_ERROR(cudaEventRecord(startToCpu, 0));
            
            // Copy from GPU to CPU
            float* host_buffer = new float[m_renderWidth * m_renderHeight * 3];
            CHECK_CUDA_ERROR(cudaMemcpy(host_buffer, source_ptr, buffer_size, cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            
            // Record and calculate the time for GPU to CPU copy
            CHECK_CUDA_ERROR(cudaEventRecord(stopToCpu, 0));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stopToCpu));
            float toCpuTimeMs = 0.0f;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&toCpuTimeMs, startToCpu, stopToCpu));
            
            // Save the GPU to CPU copy time
            if (eye == vr::Eye_Left) {
                leftEyeToCpuCopyTime = toCpuTimeMs;
            } else {
                rightEyeToCpuCopyTime = toCpuTimeMs;
            }
            
            // Add to total for this frame
            totalToCpuCopyTime += toCpuTimeMs;
            
            // Clean up events
            CHECK_CUDA_ERROR(cudaEventDestroy(startToCpu));
            CHECK_CUDA_ERROR(cudaEventDestroy(stopToCpu));
            
            // Ensure we're in the right CUDA context for resource mapping
            if (useDeviceContext) {
                if (!m_deviceContext->setDevice()) {
                    //std::cerr << "Failed to set device context for resource mapping" << std::endl;
                    useDeviceContext = false;
                    cudaSetDevice(m_deviceId);
                }
            } else {
                // Fallback to setting the device directly
                CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
            }
            
            // Create and start timing for CPU to GPU copy
            cudaEvent_t startFromCpu, stopFromCpu;
            CHECK_CUDA_ERROR(cudaEventCreate(&startFromCpu));
            CHECK_CUDA_ERROR(cudaEventCreate(&stopFromCpu));
            CHECK_CUDA_ERROR(cudaEventRecord(startFromCpu, 0));
            
            // Copy from CPU to GPU
            CHECK_CUDA_ERROR(cudaMalloc(&tensor_ptr, buffer_size));
            CHECK_CUDA_ERROR(cudaMemcpy(tensor_ptr, host_buffer, buffer_size, cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            
            // Record and calculate the time for CPU to GPU copy
            CHECK_CUDA_ERROR(cudaEventRecord(stopFromCpu, 0));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stopFromCpu));
            float fromCpuTimeMs = 0.0f;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&fromCpuTimeMs, startFromCpu, stopFromCpu));
            
            // Save the CPU to GPU copy time
            if (eye == vr::Eye_Left) {
                leftEyeFromCpuCopyTime = fromCpuTimeMs;
            } else {
                rightEyeFromCpuCopyTime = fromCpuTimeMs;
            }
            
            // Add to total for this frame
            totalFromCpuCopyTime += fromCpuTimeMs;
            
            // Clean up events
            CHECK_CUDA_ERROR(cudaEventDestroy(startFromCpu));
            CHECK_CUDA_ERROR(cudaEventDestroy(stopFromCpu));
            
            // Clean up host buffer
            delete[] host_buffer;
            //gpu_result = result.to(torch::Device(torch::kCUDA, m_deviceId));

        } else if (!result.is_cuda()) {
            gpu_result = result.to(torch::Device(torch::kCUDA, m_deviceId));
            tensor_ptr = gpu_result.data_ptr();
        } else {
            tensor_ptr = result.data_ptr(); // Already on the right device
        }


        // Create a surface to write to
        cudaResourceDesc resDesc{};
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cudaArray;

        cudaSurfaceObject_t surface;
        CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&surface, &resDesc));

        // Launch CUDA kernel to copy and convert format
        dim3 blockSize(16, 16);
        dim3 gridSize((m_renderWidth + blockSize.x - 1) / blockSize.x,
                      (m_renderHeight + blockSize.y - 1) / blockSize.y);

        // Call CUDA kernel to convert RGB float tensor to RGBA bytes for OpenGL
        convertTensorToSurface<<<gridSize, blockSize, 0, stream>>>(
            (float*)tensor_ptr,
            surface,
            m_renderWidth,
            m_renderHeight);

        // Wait for the kernel to finish using the stream
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        // Destroy surface
        CHECK_CUDA_ERROR(cudaDestroySurfaceObject(surface));

        // Unmap resource
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &eyeRes.cudaResource, stream));
        
        // Before measuring end time, make sure we're still in the right CUDA context
        if (useDeviceContext) {
            if (!m_deviceContext->setDevice()) {
                std::cerr << "Failed to set device context for ending texture copy timing" << std::endl;
                useDeviceContext = false;
                CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
                stream = 0; // Use default stream as fallback
            }
        } else {
            CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
        }
        
        // Record end event for texture copy with proper stream
        CHECK_CUDA_ERROR(cudaEventRecord(endCopy, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(endCopy));
        
        // Calculate and log the time it took to copy from OptixRenderer's buffer to VR texture
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&textureCopyTime, startCopy, endCopy));
        
        // Save texture copy time for CSV logging
        if (eye == vr::Eye_Left) {
            leftEyeTextureCopyTime = textureCopyTime;
        } else {
            rightEyeTextureCopyTime = textureCopyTime;
        }
        
        // Print to console only in debug mode with fixed precision (3 decimal places)
        DEBUG_PRINT(eyeName << " Eye - Texture copy time: " << std::fixed << std::setprecision(3) << textureCopyTime << " ms");
        
        // Clean up CUDA events in the same context they were created
        CHECK_CUDA_ERROR(cudaEventDestroy(startRender));
        CHECK_CUDA_ERROR(cudaEventDestroy(endRender));
        CHECK_CUDA_ERROR(cudaEventDestroy(startCopy));
        CHECK_CUDA_ERROR(cudaEventDestroy(endCopy));
        
        // Release the device context if used
        if (useDeviceContext) {
            m_deviceContext->releaseDevice();
        }
    } else {
        // Use the device context for fallback rendering if available
        bool useDeviceContext = m_deviceContext && m_deviceContext->isInitialized();
        CUstream stream = 0; // Default stream
        
        if (useDeviceContext) {
            // Set the device context to ensure proper CUDA context
            if (!m_deviceContext->setDevice()) {
                std::cerr << "Failed to set device context for fallback rendering" << std::endl;
                useDeviceContext = false;
            } else {
                std::cout << "Using device context for fallback rendering" << std::endl;
                stream = m_deviceContext->getStream();
            }
        } else {
            // Fallback to basic CUDA device selection
            CHECK_CUDA_ERROR(cudaSetDevice(m_deviceId));
            std::cout << "Fallback: Using cudaSetDevice for texture rendering" << std::endl;
        }
        
        // Map the texture for CUDA access
        cudaArray_t cudaArray;
        std::cout << "Mapping CUDA graphics resource for fallback rendering" << std::endl;
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &eyeRes.cudaResource, stream));
        CHECK_CUDA_ERROR(
            cudaGraphicsSubResourceGetMappedArray(&cudaArray, eyeRes.cudaResource, 0, 0));

        // Get a surface to directly write to the texture
        cudaResourceDesc resDesc{};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cudaArray;

        cudaSurfaceObject_t surface;
        CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&surface, &resDesc));

        // Use 2D memory copy to write to the array
        // Create a temporary buffer in CUDA memory
        unsigned char* deviceBuffer;
        size_t         pitch;
        CHECK_CUDA_ERROR(cudaMallocPitch(&deviceBuffer, &pitch, m_renderWidth * 4, m_renderHeight));

        // Launch the appropriate kernel to fill the buffer
        dim3 blockSize(16, 16);
        dim3 gridSize((m_renderWidth + blockSize.x - 1) / blockSize.x,
                      (m_renderHeight + blockSize.y - 1) / blockSize.y);

        if (eye == vr::Eye_Left) {
            renderLeftEyeGrid<<<gridSize, blockSize, 0, stream>>>(deviceBuffer,
                                                       m_renderWidth,
                                                       m_renderHeight,
                                                       pitch,
                                                       m_time);
        } else {
            renderRightEyeGrid<<<gridSize, blockSize, 0, stream>>>(deviceBuffer,
                                                        m_renderWidth,
                                                        m_renderHeight,
                                                        pitch,
                                                        m_time);
        }

        // Wait for the kernel to finish using the stream
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        // Copy from the buffer to the CUDA array
        CHECK_CUDA_ERROR(cudaMemcpy2DToArray(
            cudaArray, 0, 0,
            deviceBuffer, pitch,
            m_renderWidth * 4, m_renderHeight,
            cudaMemcpyDeviceToDevice
        ));

        // Free the temporary buffer
        CHECK_CUDA_ERROR(cudaFree(deviceBuffer));

        // Destroy the surface object
        CHECK_CUDA_ERROR(cudaDestroySurfaceObject(surface));

        // Unmap the resource
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &eyeRes.cudaResource, stream));
        
        // Release the device context if used
        if (useDeviceContext) {
            m_deviceContext->releaseDevice();
        }
    }
}

void OpenVRSystem::SubmitFramesToCompositor() const {
    // Submit left eye texture
    vr::Texture_t leftEyeTexture = {(void*)(uintptr_t)m_leftEye.texture, vr::TextureType_OpenGL,
                                    vr::ColorSpace_Gamma};
    vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);

    // Submit right eye texture
    vr::Texture_t rightEyeTexture = {(void*)(uintptr_t)m_rightEye.texture, vr::TextureType_OpenGL,
                                     vr::ColorSpace_Gamma};
    vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
}

void OpenVRSystem::ProcessVREvent(const vr::VREvent_t& event) {
    switch (event.eventType) {
        case vr::VREvent_Quit:
            std::cout << "Received quit event from OpenVR" << std::endl;
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            break;

        case vr::VREvent_TrackedDeviceActivated:
            std::cout << "Device " << event.trackedDeviceIndex << " activated" << std::endl;
            break;
            
        case vr::VREvent_TrackedDeviceDeactivated:
            std::cout << "Device " << event.trackedDeviceIndex << " deactivated" << std::endl;
            break;

        // Controller button events
        case vr::VREvent_ButtonPress:
        case vr::VREvent_ButtonUnpress:
        case vr::VREvent_ButtonTouch:
        case vr::VREvent_ButtonUntouch:
            // Check if it's a controller
            if (m_pVRSystem && 
                m_pVRSystem->GetTrackedDeviceClass(event.trackedDeviceIndex) == vr::TrackedDeviceClass_Controller) {
                ProcessControllerInput(event.trackedDeviceIndex, event);
            }
            break;

        default:
            break;
    }
}

bool OpenVRSystem::ShouldClose() {
    return glfwWindowShouldClose(m_window);
}

void OpenVRSystem::UpdateControllerStates() {
    if (!m_pVRSystem) return;
    
    // Array to hold poses for all devices
    vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
    
    // Get all device poses
    m_pVRSystem->GetDeviceToAbsoluteTrackingPose(
        vr::TrackingUniverseStanding, 0.0f, poses, vr::k_unMaxTrackedDeviceCount);
    
    // Find and update controller states
    for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++) {
        if (!poses[i].bDeviceIsConnected) continue;
        
        // Check device type
        if (m_pVRSystem->GetTrackedDeviceClass(i) == vr::TrackedDeviceClass_Controller) {
            // Check which hand (left or right) this controller belongs to
            vr::ETrackedControllerRole role = m_pVRSystem->GetControllerRoleForTrackedDeviceIndex(i);
            ControllerState* controller = nullptr;
            
            if (role == vr::TrackedControllerRole_LeftHand) {
                controller = &m_leftController;
            } else if (role == vr::TrackedControllerRole_RightHand) {
                controller = &m_rightController;
            } else {
                continue;  // Skip unknown controller roles
            }
            
            // Update controller validity and index
            controller->isValid = poses[i].bPoseIsValid;
            controller->deviceIndex = i;
            
            // Only update pose if it's valid
            if (poses[i].bPoseIsValid) {
                // Convert SteamVR matrix to our 3x4 format
                const vr::HmdMatrix34_t& mat = poses[i].mDeviceToAbsoluteTracking;
                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < 4; col++) {
                        controller->matrix[row * 4 + col] = mat.m[row][col];
                    }
                }
                
                // Get controller button states (only if valid)
                vr::VRControllerState_t controllerState;
                if (m_pVRSystem->GetControllerState(i, &controllerState, sizeof(controllerState))) {
                    // Trigger (analog value converted to binary press)
                    controller->triggerPressed = 
                        (controllerState.rAxis[1].x > 0.5f);  // Axis 1 is the trigger on most controllers
                    
                    // Grip button
                    controller->gripPressed = 
                        (controllerState.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_Grip)) != 0;
                    
                    // Touchpad press and position
                    controller->touchpadPressed = 
                        (controllerState.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_SteamVR_Touchpad)) != 0;
                    
                    // Touchpad coordinates (-1 to 1 range)
                    controller->touchpadX = controllerState.rAxis[0].x;
                    controller->touchpadY = controllerState.rAxis[0].y;
                }
            }
        }
    }
}

void OpenVRSystem::ProcessControllerInput(vr::TrackedDeviceIndex_t deviceIndex, const vr::VREvent_t& event) {
    // Determine which controller this event belongs to
    ControllerState* controller = nullptr;
    vr::ETrackedControllerRole role = m_pVRSystem->GetControllerRoleForTrackedDeviceIndex(deviceIndex);
    
    if (role == vr::TrackedControllerRole_LeftHand) {
        controller = &m_leftController;
    } else if (role == vr::TrackedControllerRole_RightHand) {
        controller = &m_rightController;
    } else {
        return;  // Not a known controller
    }
    
    // Update controller state based on event
    switch (event.eventType) {
        case vr::VREvent_ButtonPress:
            if (event.data.controller.button == vr::k_EButton_SteamVR_Trigger) {
                controller->triggerPressed = true;
                std::cout << (role == vr::TrackedControllerRole_LeftHand ? "Left" : "Right") 
                         << " controller trigger pressed" << std::endl;
            } else if (event.data.controller.button == vr::k_EButton_Grip) {
                controller->gripPressed = true;
                std::cout << (role == vr::TrackedControllerRole_LeftHand ? "Left" : "Right") 
                         << " controller grip pressed" << std::endl;
            } else if (event.data.controller.button == vr::k_EButton_SteamVR_Touchpad) {
                controller->touchpadPressed = true;
                std::cout << (role == vr::TrackedControllerRole_LeftHand ? "Left" : "Right") 
                         << " controller touchpad pressed" << std::endl;
            }
            break;
            
        case vr::VREvent_ButtonUnpress:
            if (event.data.controller.button == vr::k_EButton_SteamVR_Trigger) {
                controller->triggerPressed = false;
            } else if (event.data.controller.button == vr::k_EButton_Grip) {
                controller->gripPressed = false;
            } else if (event.data.controller.button == vr::k_EButton_SteamVR_Touchpad) {
                controller->touchpadPressed = false;
            }
            break;
    }
}

void OpenVRSystem::PollEvents() {
    glfwPollEvents();

    // Check for escape key to exit
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }
    
    // Update controller states - ensure we always have the latest controller state
    UpdateControllerStates();
    
    // Poll for VR events
    vr::VREvent_t event;
    while (m_pVRSystem && m_pVRSystem->PollNextEvent(&event, sizeof(event))) {
        ProcessVREvent(event);
    }
    
    // Update controller states one more time after processing events to ensure 
    // we have the latest state after all events have been processed
    UpdateControllerStates();
}

void OpenVRSystem::SetRenderers(optix_renderer::Renderer* leftRenderer,
                                optix_renderer::Renderer* rightRenderer) {
    m_leftEyeRenderer  = leftRenderer;
    m_rightEyeRenderer = rightRenderer;
}

std::pair<uint32_t, uint32_t> OpenVRSystem::GetRecommendedRenderSize() const {
    if (m_pVRSystem) {
        return {m_renderWidth, m_renderHeight};
    }
    return {1024, 1024};// Default if not initialized
}

// Helper function to invert a rotation matrix (just the transpose for rotation matrices)
void InvertRotationMatrix(const float rot[3][3], float invRot[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            invRot[i][j] = rot[j][i]; // Transpose
        }
    }
}

// Helper function to convert OpenVR matrix to camera view matrix
void ConvertToViewMatrix(const vr::HmdMatrix34_t& openVrMatrix, float* viewMatrix) {
    // OpenVR matrices are row-major, but we need to convert to a specific format
    // First extract the rotation part (3x3)
    float rotationMatrix[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rotationMatrix[i][j] = openVrMatrix.m[i][j];
        }
    }
    
    // Invert the rotation matrix (transpose for rotation matrices)
    float invRotation[3][3];
    InvertRotationMatrix(rotationMatrix, invRotation);
    
    // Extract translation
    float translation[3] = {
        openVrMatrix.m[0][3],
        openVrMatrix.m[1][3],
        openVrMatrix.m[2][3]
    };
    
    // Negate and rotate the translation vector
    float negRotatedTranslation[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            negRotatedTranslation[i] -= invRotation[i][j] * translation[j];
        }
    }
    
    // Build the final view matrix
    // First set rotation part
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            viewMatrix[i * 4 + j] = invRotation[i][j];
        }
    }
    
    // Then set translation part
    for (int i = 0; i < 3; i++) {
        viewMatrix[i * 4 + 3] = negRotatedTranslation[i];
    }
}

void OpenVRSystem::GetEyeTransforms(float* leftEyeMatrix, float* rightEyeMatrix) {
    if (!m_pVRSystem)
        return;
    
    // Get current poses for all tracked devices (HMD and controllers)
    vr::TrackedDevicePose_t trackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
    vr::TrackedDevicePose_t gamePoses[vr::k_unMaxTrackedDeviceCount];
    
    // Get the latest pose data
    vr::VRCompositor()->WaitGetPoses(trackedDevicePoses, 
                                     vr::k_unMaxTrackedDeviceCount,
                                     gamePoses,
                                     vr::k_unMaxTrackedDeviceCount);
    
    // Check if we have a valid pose
    if (!trackedDevicePoses[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
        // Use identity transform if pose is not valid
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                leftEyeMatrix[i * 4 + j] = (i == j) ? 1.0f : 0.0f;
                rightEyeMatrix[i * 4 + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        return;
    }
    
    // Get the head pose matrix - this maps from tracking space to head space
    const vr::HmdMatrix34_t& headPose = trackedDevicePoses[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking;
    
    // Get the eye poses - these map from head space to eye space
    vr::HmdMatrix34_t leftEyePose = m_pVRSystem->GetEyeToHeadTransform(vr::Eye_Left);
    vr::HmdMatrix34_t rightEyePose = m_pVRSystem->GetEyeToHeadTransform(vr::Eye_Right);
    
    // IMPORTANT: In OpenVR, GetEyeToHeadTransform returns the transform FROM eye TO head space
    // This is actually the inverse of what we want, which is the transform FROM head TO eye space
    // So we need to be careful with the matrix composition
    
    // For correct eye transforms, what we need is: 
    // worldToEye = worldToHead * headToEye = worldToHead * inverse(eyeToHead)
    
    // For simplicity, we'll use the fact that these are rigid transforms with no scale,
    // so the inverse of the rotation part is just the transpose. And the translation
    // needs to be rotated by the inverse rotation and negated.
    
    // First, let's create the left eye transform
    vr::HmdMatrix34_t leftEyeWorld = {};
    
    // 1. Copy the head transform as our starting point
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            leftEyeWorld.m[i][j] = headPose.m[i][j];
        }
    }
    
    // 2. Apply the eye offset
    // Since eyeToHead gives us the offset from eye to head,
    // we need to negate this offset and apply the head's rotation to get the world-space offset
    float eyeOffsetInWorldSpace[3] = {0};
    
    // First negate the eye offset (to get the head-to-eye vector)
    float eyeOffsetX = -leftEyePose.m[0][3];
    float eyeOffsetY = -leftEyePose.m[1][3];
    float eyeOffsetZ = -leftEyePose.m[2][3];
    
    // Now rotate this offset by the head's rotation matrix
    // to transform it into world space
    for (int i = 0; i < 3; i++) {
        eyeOffsetInWorldSpace[i] = 
            headPose.m[i][0] * eyeOffsetX +
            headPose.m[i][1] * eyeOffsetY +
            headPose.m[i][2] * eyeOffsetZ;
    }
    
    // Apply this offset to the head position
    leftEyeWorld.m[0][3] += eyeOffsetInWorldSpace[0];
    leftEyeWorld.m[1][3] += eyeOffsetInWorldSpace[1];
    leftEyeWorld.m[2][3] += eyeOffsetInWorldSpace[2];
    
    // Now do the same for the right eye
    vr::HmdMatrix34_t rightEyeWorld = {};
    
    // 1. Copy the head transform
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            rightEyeWorld.m[i][j] = headPose.m[i][j];
        }
    }
    
    // 2. Apply the eye offset
    // First negate the eye offset (to get the head-to-eye vector)
    eyeOffsetX = -rightEyePose.m[0][3];
    eyeOffsetY = -rightEyePose.m[1][3];
    eyeOffsetZ = -rightEyePose.m[2][3];
    
    // Clear previous values
    memset(eyeOffsetInWorldSpace, 0, sizeof(eyeOffsetInWorldSpace));
    
    // Now rotate this offset by the head's rotation matrix
    for (int i = 0; i < 3; i++) {
        eyeOffsetInWorldSpace[i] = 
            headPose.m[i][0] * eyeOffsetX +
            headPose.m[i][1] * eyeOffsetY +
            headPose.m[i][2] * eyeOffsetZ;
    }
    
    // Apply this offset to the head position
    rightEyeWorld.m[0][3] += eyeOffsetInWorldSpace[0];
    rightEyeWorld.m[1][3] += eyeOffsetInWorldSpace[1];
    rightEyeWorld.m[2][3] += eyeOffsetInWorldSpace[2];
    
    // Copy the transformed matrices directly
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            leftEyeMatrix[i * 4 + j] = leftEyeWorld.m[i][j];
            rightEyeMatrix[i * 4 + j] = rightEyeWorld.m[i][j];
        }
    }
    
    // Print debug info occasionally
    static int call_count = 0;
    if (call_count++ % 30 == 0) {
        // Print head position
        DEBUG_PRINT("Head Position: " 
                  << headPose.m[0][3] << ", " 
                  << headPose.m[1][3] << ", " 
                  << headPose.m[2][3]);
                  
        // Print final eye positions
        DEBUG_PRINT("Final Left Eye Position: " 
                  << leftEyeMatrix[3] << ", " 
                  << leftEyeMatrix[7] << ", " 
                  << leftEyeMatrix[11]);
        
        DEBUG_PRINT("Final Right Eye Position: " 
                  << rightEyeMatrix[3] << ", " 
                  << rightEyeMatrix[7] << ", " 
                  << rightEyeMatrix[11]);
    }
}

// New function to return raw eye-to-head transforms
void OpenVRSystem::GetEyeToHeadTransforms(float* leftEyeToHeadMatrix, float* rightEyeToHeadMatrix) {
    if (!m_pVRSystem)
        return;
        
    // Get the eye-to-head transforms directly from OpenVR
    vr::HmdMatrix34_t leftEyeToHead = m_pVRSystem->GetEyeToHeadTransform(vr::Eye_Left);
    vr::HmdMatrix34_t rightEyeToHead = m_pVRSystem->GetEyeToHeadTransform(vr::Eye_Right);
    
    // Copy the data to the output arrays (3x4 matrices)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            leftEyeToHeadMatrix[i * 4 + j] = leftEyeToHead.m[i][j];
            rightEyeToHeadMatrix[i * 4 + j] = rightEyeToHead.m[i][j];
        }
    }
    
    // Print debug info occasionally
    static int call_count = 0;
    if (call_count++ % 30 == 0) {
        // Print eye offsets 
        DEBUG_PRINT("Left Eye Translation (from eye to head): " 
                  << leftEyeToHead.m[0][3] << ", " 
                  << leftEyeToHead.m[1][3] << ", " 
                  << leftEyeToHead.m[2][3]);
                  
        DEBUG_PRINT("Right Eye Translation (from eye to head): " 
                  << rightEyeToHead.m[0][3] << ", " 
                  << rightEyeToHead.m[1][3] << ", " 
                  << rightEyeToHead.m[2][3]);
    }
}

// New function to return raw head pose matrix
void OpenVRSystem::GetHeadPoseMatrix(float* headPoseMatrix) {
    if (!m_pVRSystem)
        return;
        
    // Get current tracking poses
    vr::TrackedDevicePose_t trackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
    vr::TrackedDevicePose_t gamePoses[vr::k_unMaxTrackedDeviceCount];
    
    // Get the latest pose data
    vr::VRCompositor()->WaitGetPoses(trackedDevicePoses, 
                                     vr::k_unMaxTrackedDeviceCount,
                                     gamePoses,
                                     vr::k_unMaxTrackedDeviceCount);
    
    // Check if we have a valid pose
    if (!trackedDevicePoses[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
        // Use identity transform if pose is not valid
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                headPoseMatrix[i * 4 + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        return;
    }
    
    // Get the head pose matrix - this maps from tracking space to head space
    const vr::HmdMatrix34_t& headPose = trackedDevicePoses[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking;


    
    // Copy the matrix data directly (3x4 matrix)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            headPoseMatrix[i * 4 + j] = headPose.m[i][j];
        }
    }
    
    // Print debug info occasionally
    static int call_count = 0;
    if (call_count++ % 30 == 0) {
        // Print head pose information
        DEBUG_PRINT("Head Position (from head pose matrix): " 
                  << headPose.m[0][3] << ", " 
                  << headPose.m[1][3] << ", " 
                  << headPose.m[2][3]);
                  
        // Print rotation part (first column)
        DEBUG_PRINT("Head Orientation (X axis): " 
                  << headPose.m[0][0] << ", " 
                  << headPose.m[1][0] << ", " 
                  << headPose.m[2][0]);
    }
}

void OpenVRSystem::GetProjectionMatrices(float* leftProjMatrix,
                                         float* rightProjMatrix,
                                         float  nearClip,
                                         float  farClip) {
    if (!m_pVRSystem)
        return;

    vr::HmdMatrix44_t leftMat = m_pVRSystem->GetProjectionMatrix(
        vr::Eye_Left,
        nearClip,
        farClip);
    vr::HmdMatrix44_t rightMat = m_pVRSystem->GetProjectionMatrix(
        vr::Eye_Right,
        nearClip,
        farClip);

    // Copy data to provided arrays (4x4 matrices)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            leftProjMatrix[i * 4 + j]  = leftMat.m[i][j];
            rightProjMatrix[i * 4 + j] = rightMat.m[i][j];
        }
    }
}

// Controller accessor methods

bool OpenVRSystem::IsControllerValid(bool isLeft) const {
    return isLeft ? m_leftController.isValid : m_rightController.isValid;
}

void OpenVRSystem::GetControllerPose(bool isLeft, float* poseMatrix) {
    const ControllerState& controller = isLeft ? m_leftController : m_rightController;
    
    if (!controller.isValid) {
        // Return identity matrix if controller is not valid
        for (int i = 0; i < 12; i++) {
            poseMatrix[i] = (i % 5 == 0) ? 1.0f : 0.0f;  // Identity matrix in column-major format
        }
        return;
    }
    
    // Copy the controller's pose matrix
    for (int i = 0; i < 12; i++) {
        poseMatrix[i] = controller.matrix[i];
    }
}

bool OpenVRSystem::IsControllerTriggerPressed(bool isLeft) const {
    const ControllerState& controller = isLeft ? m_leftController : m_rightController;
    return controller.isValid && controller.triggerPressed;
}

bool OpenVRSystem::IsControllerGripPressed(bool isLeft) const {
    const ControllerState& controller = isLeft ? m_leftController : m_rightController;
    return controller.isValid && controller.gripPressed;
}

void OpenVRSystem::GetControllerTouchpadPosition(bool isLeft, float* x, float* y) {
    const ControllerState& controller = isLeft ? m_leftController : m_rightController;
    
    if (controller.isValid) {
        *x = controller.touchpadX;
        *y = controller.touchpadY;
    } else {
        *x = 0.0f;
        *y = 0.0f;
    }
}

bool OpenVRSystem::IsControllerTouchpadPressed(bool isLeft) const {
    const ControllerState& controller = isLeft ? m_leftController : m_rightController;
    return controller.isValid && controller.touchpadPressed;
}

// Method to move a scene object using controller
void OpenVRSystem::MoveObjectWithController(optix_renderer::GeometryInstance* instance, bool isLeft) {
    if (!instance || !m_controllerInteractionEnabled) return;
    
    const ControllerState& controller = isLeft ? m_leftController : m_rightController;
    
    // Only proceed if controller is valid and trigger is pressed (to "grab" the object)
    if (!controller.isValid || !controller.triggerPressed) return;
    
    // Extract position from controller's pose matrix and create a vector for it
    std::vector<float> position = {
        controller.matrix[3],   // X - 4th element of first row
        controller.matrix[7],   // Y - 4th element of second row
        controller.matrix[11]   // Z - 4th element of third row
    };
    
    // Extract rotation as Euler angles (simplified version) and create a vector
    std::vector<float> rotation(3, 0.0f);
    
    // A simple method to extract approximate Euler angles from rotation matrix
    rotation[1] = atan2(-controller.matrix[8], sqrtf(controller.matrix[0]*controller.matrix[0] + 
                                                   controller.matrix[4]*controller.matrix[4])); // pitch
    
    if (fabs(rotation[1]) > 0.998f) {
        // Near gimbal lock
        rotation[0] = atan2(-controller.matrix[1], controller.matrix[5]); // roll
        rotation[2] = 0; // yaw
    } else {
        rotation[0] = atan2(controller.matrix[9], controller.matrix[10]); // roll
        rotation[2] = atan2(controller.matrix[4], controller.matrix[0]);  // yaw
    }
    
    // Use fixed scale vector
    std::vector<float> scale = {1.0f, 1.0f, 1.0f};
    
    // Apply the transformation to the instance using setTransform method
    instance->setTransform(position, rotation, scale);
}

float OpenVRSystem::GetLeftEyeRenderTimeMs() const { return leftEyeRenderTime; }
float OpenVRSystem::GetLeftEyeInternalCopyTimeMs() const { return leftEyeInternalCopyTime; }
float OpenVRSystem::GetLeftEyeTextureCopyTimeMs() const { return leftEyeTextureCopyTime; }
float OpenVRSystem::GetLeftEyeToCpuCopyTimeMs() const { return leftEyeToCpuCopyTime; }
float OpenVRSystem::GetLeftEyeFromCpuCopyTimeMs() const { return leftEyeFromCpuCopyTime; }

float OpenVRSystem::GetRightEyeRenderTimeMs() const { return rightEyeRenderTime; }
float OpenVRSystem::GetRightEyeInternalCopyTimeMs() const { return rightEyeInternalCopyTime; }
float OpenVRSystem::GetRightEyeTextureCopyTimeMs() const { return rightEyeTextureCopyTime; }
float OpenVRSystem::GetRightEyeToCpuCopyTimeMs() const { return rightEyeToCpuCopyTime; }
float OpenVRSystem::GetRightEyeFromCpuCopyTimeMs() const { return rightEyeFromCpuCopyTime; }

float OpenVRSystem::GetTotalToCpuCopyTimeMs() const { return totalToCpuCopyTime; }
float OpenVRSystem::GetTotalFromCpuCopyTimeMs() const { return totalFromCpuCopyTime; }

torch::Tensor OpenVRSystem::GetLastLeftEyeCPU() {
    // Return a clone to ensure Python gets its own copy and to manage lifetime,
    // though m_lastLeftEyeTensor_CPU is already a CPU clone.
    // If m_lastLeftEyeTensor_CPU is not yet initialized, return an empty tensor.
    if (m_lastLeftEyeTensor_CPU.defined()) {
        return m_lastLeftEyeTensor_CPU.clone();
    }
    return torch::Tensor(); // Return empty tensor if not ready
}

torch::Tensor OpenVRSystem::GetLastRightEyeCPU() {
    if (m_lastRightEyeTensor_CPU.defined()) {
        return m_lastRightEyeTensor_CPU.clone();
    }
    return torch::Tensor();
}
/*
int main(int argc, char* argv[]) {
    // Initialize CUDA
    const cudaError_t cudaStatus = cudaFree(nullptr);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA initialization failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // Create OpenVR system
    OpenVRSystem vrSystem;
    if (!vrSystem.Initialize()) {
        std::cerr << "Failed to initialize OpenVR system" << std::endl;
        return -1;
    }

    std::cout << "Running main loop... Press ESC to exit." << std::endl;

    // Main loop
    while (!vrSystem.ShouldClose())
    {
        // Render a frame
        vrSystem.RenderFrame();

        // Poll for window events
        vrSystem.PollEvents();
        std::cout<<"pool"<<std::endl;

        // Sleep briefly to prevent 100% CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::cout<<"sleep"<<std::endl;
    }

    std::cout << "Shutting down..." << std::endl;

    return 0;
}*/