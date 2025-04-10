//
// Created by artem on 4/7/25.
//

#include "../include/OpenVRSystem.cuh"

// CUDA kernel for rendering a grid pattern to the left eye (red and black)
__global__ void renderLeftEyeGrid(unsigned char* surface, int width, int height, int pitch, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate output position in the surface
    unsigned char* pixel = surface + y * pitch + x * 4;

    // Create a grid pattern that moves slowly over time
    int gridSize = 32; // Size of each grid cell
    int offsetX = (int)(time * 10.0f) % gridSize;
    int offsetY = (int)(time * 5.0f) % gridSize;

    bool isGridLine = ((x + offsetX) % gridSize < 2) || ((y + offsetY) % gridSize < 2);

    if (isGridLine) {
        // Grid lines: Red
        pixel[0] = 0;     // B
        pixel[1] = 0;     // G
        pixel[2] = 255;   // R
        pixel[3] = 255;   // A
    } else {
        // Grid cells: Black
        pixel[0] = 0;     // B
        pixel[1] = 0;     // G
        pixel[2] = 0;     // R
        pixel[3] = 255;   // A
    }
}

// CUDA kernel for rendering a grid pattern to the right eye (blue and black)
__global__ void renderRightEyeGrid(unsigned char* surface, int width, int height, int pitch, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate output position in the surface
    unsigned char* pixel = surface + y * pitch + x * 4;

    // Create a grid pattern that moves slowly over time (different direction from left eye)
    int gridSize = 32; // Size of each grid cell
    int offsetX = (int)(time * 5.0f) % gridSize;
    int offsetY = (int)(time * 10.0f) % gridSize;

    bool isGridLine = ((x + offsetX) % gridSize < 2) || ((y + offsetY) % gridSize < 2);

    if (isGridLine) {
        // Grid lines: Blue
        pixel[0] = 255;   // B
        pixel[1] = 0;     // G
        pixel[2] = 0;     // R
        pixel[3] = 255;   // A
    } else {
        // Grid cells: Black
        pixel[0] = 0;     // B
        pixel[1] = 0;     // G
        pixel[2] = 0;     // R
        pixel[3] = 255;   // A
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
        m_pVRSystem = vr::VR_Init(&eError, vr::VRApplication_Scene);
        if (eError != vr::VRInitError_None) {
            std::cerr << "Failed to initialize OpenVR: "
                      << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;
            return false;
        }

        std::cout << "OpenVR initialized successfully" << std::endl;

        // Get the recommended render target size
        m_pVRSystem->GetRecommendedRenderTargetSize(&m_renderWidth, &m_renderHeight);
        std::cout << "Recommended render target size: " << m_renderWidth << "x" << m_renderHeight << std::endl;

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

        return true;
    }

    void OpenVRSystem::Shutdown() {
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_renderWidth, m_renderHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

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
            return false;
        }

        return true;
    }

    void OpenVRSystem::RenderFrame() {
        // Update time
        m_time += 0.01f;
        std::cout << "pool 1" << std::endl;
        // Process VR events
        vr::VREvent_t event{};
        while (m_pVRSystem->PollNextEvent(&event, sizeof(event))) {
            ProcessVREvent(event);
        }
        std::cout << "pool 2" << std::endl;
        std::cout << "pose 1" << std::endl;
        // Wait for poses to synchronize with the display
        vr::TrackedDevicePose_t trackedDevicePoses[vr::k_unMaxTrackedDeviceCount];
        vr::TrackedDevicePose_t gamePoses[vr::k_unMaxTrackedDeviceCount];  // Add this
        std::cout << "pose 2" << std::endl;
        std::cout << "compositor 1" << std::endl;
        vr::VRCompositor()->WaitGetPoses(trackedDevicePoses, vr::k_unMaxTrackedDeviceCount, gamePoses, vr::k_unMaxTrackedDeviceCount);
        std::cout << "compositor 2" << std::endl;

        std::cout<<"lef1"<<std::endl;
        // Render to the left eye texture using CUDA
        RenderEyeTexture(vr::Eye_Left);
        std::cout<<"lef2"<<std::endl;

        std::cout<<"right1"<<std::endl;
        // Render to the right eye texture using CUDA
        RenderEyeTexture(vr::Eye_Right);
        std::cout<<"right2"<<std::endl;


        // Submit the textures to the compositor
        SubmitFramesToCompositor();
        std::cout<<"submitted"<<std::endl;
    }

    void OpenVRSystem::RenderEyeTexture(vr::EVREye eye) {
        EyeResources& eyeRes = (eye == vr::Eye_Left) ? m_leftEye : m_rightEye;

        // Map the texture for CUDA access
        cudaArray_t cudaArray;
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &eyeRes.cudaResource));
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaArray, eyeRes.cudaResource, 0, 0));

        // Get a surface to directly write to the texture
        cudaResourceDesc resDesc{};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaArray;

        cudaSurfaceObject_t surface;
        CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&surface, &resDesc));

        // Use 2D memory copy to write to the array
        // Create a temporary buffer in CUDA memory
        unsigned char* deviceBuffer;
        size_t pitch;
        CHECK_CUDA_ERROR(cudaMallocPitch(&deviceBuffer, &pitch, m_renderWidth * 4, m_renderHeight));

        // Launch the appropriate kernel to fill the buffer
        dim3 blockSize(16, 16);
        dim3 gridSize((m_renderWidth + blockSize.x - 1) / blockSize.x,
                      (m_renderHeight + blockSize.y - 1) / blockSize.y);

        if (eye == vr::Eye_Left) {
            renderLeftEyeGrid<<<gridSize, blockSize>>>(deviceBuffer, m_renderWidth, m_renderHeight, pitch, m_time);
        } else {
            renderRightEyeGrid<<<gridSize, blockSize>>>(deviceBuffer, m_renderWidth, m_renderHeight, pitch, m_time);
        }

        // Wait for the kernel to finish
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

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
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &eyeRes.cudaResource));
    }

    void OpenVRSystem::SubmitFramesToCompositor() const {
        // Submit left eye texture
        vr::Texture_t leftEyeTexture = {(void*)(uintptr_t)m_leftEye.texture, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
        vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);

        // Submit right eye texture
        vr::Texture_t rightEyeTexture = {(void*)(uintptr_t)m_rightEye.texture, vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
        vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
    }

    void OpenVRSystem::ProcessVREvent(const vr::VREvent_t& event) {
        switch (event.eventType) {
            case vr::VREvent_Quit:
                std::cout << "Received quit event from OpenVR" << std::endl;
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
                break;

            case vr::VREvent_ButtonPress:
                std::cout << "Button press on device " << event.trackedDeviceIndex << std::endl;
                break;

            default:
                break;
        }
    }

    bool OpenVRSystem::ShouldClose() {
        return glfwWindowShouldClose(m_window);
    }

    void OpenVRSystem::PollEvents() {
        glfwPollEvents();

        // Check for escape key to exit
        if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        }
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