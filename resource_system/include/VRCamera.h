
#pragma once

#include <array>
#include <vector>

#include "Renderer.h"  // This includes the CameraParameters definition


namespace ors {

struct CameraSetupParams {
    std::vector<float>   eyeMatrix;        // 3x4 matrix (currently unused)
    std::vector<float>   eyeToHeadMatrix;  // 3x4 matrix
    std::vector<float>   headPoseMatrix;   // 3x4 matrix
    std::vector<float>   projMatrix;       // 4x4 matrix
    std::array<float, 3> basePos;          // Base position for locked camera
    bool                 lockPosition;     // Whether to lock camera position
};

optix_renderer::CameraParameters setupVRCamera(const CameraSetupParams& params);

} // namespace ors