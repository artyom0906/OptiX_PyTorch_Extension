//
// Created by artem on 4/7/25.
//
#pragma once
#include <openvr.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>  // For std::fixed and std::setprecision

//#define DEBUG
// Debug printing macro
#ifdef DEBUG
#define DEBUG_PRINT(x) std::cout << x << std::endl
#else
#define DEBUG_PRINT(x)
#endif

// Define this macro to include "ms" units after each timing value in CSV
// Comment out to exclude units for cleaner data import
#define CSV_INCLUDE_UNITS

#include "Renderer.h"
#include "GeometryInstance.h"

// Structure to hold eye-specific rendering resources
struct EyeResources {
    GLuint texture;                 // OpenGL texture
    cudaGraphicsResource_t cudaResource; // CUDA resource for the texture

    EyeResources() : texture(0), cudaResource(nullptr) {}
};

// Controller state structure to track VR controllers
struct ControllerState {
    bool isValid;                // Is the controller active/valid
    vr::TrackedDeviceIndex_t deviceIndex; // OpenVR device index
    float matrix[12];           // 3x4 pose matrix (position and orientation)
    bool triggerPressed;        // Trigger button state
    bool gripPressed;           // Grip button state
    bool touchpadPressed;       // Touchpad pressed state
    float touchpadX;            // Touchpad X position (-1 to 1)
    float touchpadY;            // Touchpad Y position (-1 to 1)
    
    ControllerState() : isValid(false), deviceIndex(vr::k_unTrackedDeviceIndexInvalid),
                      triggerPressed(false), gripPressed(false), 
                      touchpadPressed(false), touchpadX(0.0f), touchpadY(0.0f) {
        // Initialize matrix to identity
        for (int i = 0; i < 12; i++) {
            matrix[i] = (i % 5 == 0) ? 1.0f : 0.0f; // Identity matrix in column-major format
        }
    }
};

// OpenVR system wrapper
class OpenVRSystem {
private:
    vr::IVRSystem* m_pVRSystem;
    uint32_t m_renderWidth;
    uint32_t m_renderHeight;
    EyeResources m_leftEye;
    EyeResources m_rightEye;
    GLFWwindow* m_window;
    float m_time;

    optix_renderer::Renderer* m_leftEyeRenderer;
    optix_renderer::Renderer* m_rightEyeRenderer;
    optix_renderer::ResourceManager* m_resourceManager;
    optix_renderer::DeviceContext* m_deviceContext;  // Device context for CUDA operations
    int m_deviceId;                                 // The device ID used for VR rendering
    
    // Controller state
    ControllerState m_leftController;               // Left controller state
    ControllerState m_rightController;              // Right controller state
    bool m_controllerInteractionEnabled;            // Whether controller interaction is enabled

    torch::Tensor m_lastLeftEyeTensor_CPU;
    torch::Tensor m_lastRightEyeTensor_CPU;
    
    // Performance logging
    std::ofstream m_perfLogFile;                    // Output file for performance metrics
    unsigned int m_frameCount;                      // Frame counter
    std::string m_perfLogFilename;                  // Filename for performance log
    
    // Helper method to initialize performance logging
    bool InitPerfLogging();
    
    // Helper methods for controller tracking
    void UpdateControllerStates();
    void ProcessControllerInput(vr::TrackedDeviceIndex_t deviceIndex, const vr::VREvent_t& event);


public:
    OpenVRSystem(optix_renderer::ResourceManager* resourceManager = nullptr, int deviceId = 0) 
        : m_pVRSystem(nullptr), m_renderWidth(0), m_renderHeight(0), m_window(nullptr), m_time(0.0f), 
          m_leftEyeRenderer(nullptr), m_rightEyeRenderer(nullptr), 
          m_resourceManager(resourceManager), m_deviceContext(nullptr), m_deviceId(deviceId),
          m_controllerInteractionEnabled(true), m_frameCount(0) {
        
        // Initialize device context if resource manager is provided
        if (m_resourceManager) {
            m_deviceContext = m_resourceManager->getDeviceContext(m_deviceId);
        }
        
        // Generate unique filename with timestamp for performance log
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&time_t_now));
        m_perfLogFilename = "vr_perf_" + std::string(timestamp) + ".csv";
    }

    ~OpenVRSystem() {
        Shutdown();
    }

    torch::Tensor GetLastLeftEyeCPU();
    torch::Tensor GetLastRightEyeCPU();

    float GetLeftEyeRenderTimeMs() const;
    float GetLeftEyeInternalCopyTimeMs() const;
    float GetLeftEyeTextureCopyTimeMs() const;
    float GetLeftEyeToCpuCopyTimeMs() const;
    float GetLeftEyeFromCpuCopyTimeMs() const;

    float GetRightEyeRenderTimeMs() const;
    float GetRightEyeInternalCopyTimeMs() const;
    float GetRightEyeTextureCopyTimeMs() const;
    float GetRightEyeToCpuCopyTimeMs() const;
    float GetRightEyeFromCpuCopyTimeMs() const;

    float GetTotalToCpuCopyTimeMs() const;   // For overall frame debug
    float GetTotalFromCpuCopyTimeMs() const; // For overall frame debug

    bool Initialize();
    void Shutdown();
    void RenderFrame();
    void RenderEyeTexture(vr::EVREye eye);
    void SubmitFramesToCompositor() const;
    bool CreateTextureForEye(EyeResources& eye) const;

    void ProcessVREvent(const vr::VREvent_t& event);
    bool ShouldClose();
    void PollEvents();

    void SetRenderers(optix_renderer::Renderer* leftRenderer,
                      optix_renderer::Renderer* rightRenderer);

    // Method to get recommended render resolution
    std::pair<uint32_t, uint32_t> GetRecommendedRenderSize() const;

    // Get raw eye transforms and tracking poses to process in Python
    void GetEyeTransforms(float* leftEyeMatrix, float* rightEyeMatrix);
    void GetEyeToHeadTransforms(float* leftEyeToHeadMatrix, float* rightEyeToHeadMatrix);
    void GetHeadPoseMatrix(float* headPoseMatrix);
    void GetProjectionMatrices(float* leftProjMatrix, float* rightProjMatrix,
                               float nearClip = 0.1f, float farClip = 100.0f);
                               
    // Controller-related methods
    bool IsControllerValid(bool isLeft) const;
    void GetControllerPose(bool isLeft, float* poseMatrix);
    bool IsControllerTriggerPressed(bool isLeft) const;
    bool IsControllerGripPressed(bool isLeft) const;
    void GetControllerTouchpadPosition(bool isLeft, float* x, float* y);
    bool IsControllerTouchpadPressed(bool isLeft) const;
    
    // Enable/disable controller interaction
    void SetControllerInteractionEnabled(bool enabled) { m_controllerInteractionEnabled = enabled; }
    bool IsControllerInteractionEnabled() const { return m_controllerInteractionEnabled; }
    
    // Method to move a scene object using controller
    void MoveObjectWithController(optix_renderer::GeometryInstance* instance, bool isLeft);


};