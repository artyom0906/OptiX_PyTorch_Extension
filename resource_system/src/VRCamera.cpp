
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include "../include/VRCamera.h"

namespace ors {

optix_renderer::CameraParameters setupVRCamera(const CameraSetupParams& params) {
    optix_renderer::CameraParameters cam;
    
    // Convert matrices to Eigen
    Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> headPoseNp(params.headPoseMatrix.data());
    Eigen::Map<const Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> eyeToHeadNp(params.eyeToHeadMatrix.data());
    Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> projNp(params.projMatrix.data());

    // Calculate position
    Eigen::Vector3f position;
    if (params.lockPosition) {
        position = Eigen::Map<const Eigen::Vector3f>(params.basePos.data());
    } else {
        Eigen::Vector3f eyeOffset = eyeToHeadNp.block<3,1>(0,3);
        Eigen::Matrix4f hp4 = Eigen::Matrix4f::Identity();
        hp4.block<3,4>(0,0) = headPoseNp;
        
        Eigen::Vector4f homogeneous;
        homogeneous << eyeOffset, 1.0f;
        position = (hp4 * homogeneous).head<3>();
        position.z() *= -1.0f;
    }

    // Calculate orientation using rotation matrix
    Eigen::Matrix3f rotMatrix = headPoseNp.block<3,3>(0,0);
    Eigen::Vector3f euler = rotMatrix.eulerAngles(0,1,2); // xyz order

    // Invert pitch and yaw
    euler[0] = -euler[0];  // pitch
    euler[1] = -euler[1];  // yaw
    
    // Create rotation matrix from adjusted angles
    Eigen::Matrix3f adjustedRot;
    adjustedRot = Eigen::AngleAxisf(euler[2], Eigen::Vector3f::UnitZ()) *
                  Eigen::AngleAxisf(euler[1], Eigen::Vector3f::UnitY()) *
                  Eigen::AngleAxisf(euler[0], Eigen::Vector3f::UnitX());

    // Calculate camera basis vectors
    Eigen::Vector3f right = adjustedRot * Eigen::Vector3f(1,0,0);
    Eigen::Vector3f up = adjustedRot * Eigen::Vector3f(0,1,0);
    Eigen::Vector3f forward = adjustedRot * Eigen::Vector3f(0,0,-1);

    // Calculate projection parameters
    float tanHalfW = 1.0f / std::abs(projNp(0,0));
    float tanHalfH = 1.0f / std::abs(projNp(1,1));
    float offX = projNp(0,2);
    float offY = projNp(1,2);

    // Calculate camera vectors
    Eigen::Vector3f cameraU = right;
    Eigen::Vector3f cameraV = up;
    Eigen::Vector3f cameraW = position - 
                             tanHalfW * (offX + 1.0f) * right - 
                             tanHalfH * (offY + 1.0f) * up - 
                             forward;

    // Copy results to camera parameters
    cam.position = {position[0], position[1], position[2]};
    cam.u = {cameraU[0], cameraU[1], cameraU[2]};
    cam.v = {cameraV[0], cameraV[1], cameraV[2]};
    cam.w = {cameraW[0], cameraW[1], cameraW[2]};


    return cam;
}

} // namespace ors