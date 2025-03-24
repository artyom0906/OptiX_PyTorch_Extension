#pragma once

#include <torch/extension.h>
#include "ResourceTypes.h"

namespace optix_renderer {

// Forward declarations
class ResourceManager;

// Represents an instance of geometry with material and transform
class GeometryInstance {
public:
    GeometryInstance(ResourceManager* resourceManager, 
                   GeometryHandle geometryHandle, 
                   MaterialHandle materialHandle,
                   const torch::Tensor& transform = torch::eye(4))
        : m_resourceManager(resourceManager)
        , m_geometryHandle(geometryHandle)
        , m_materialHandle(materialHandle)
        , m_transform(transform.clone())
        , m_visible(true)
    {
        // Ensure transform is a 4x4 matrix in the correct format
        AT_ASSERT(m_transform.sizes() == torch::IntArrayRef({4, 4}), "Transform must be a 4x4 matrix");
        if (m_transform.scalar_type() != torch::kFloat32) {
            m_transform = m_transform.to(torch::kFloat32);
        }
    }

    // Accessors
    GeometryHandle getGeometryHandle() const { return m_geometryHandle; }
    MaterialHandle getMaterialHandle() const { return m_materialHandle; }
    
    const torch::Tensor& getTransform() const { return m_transform; }
    void setTransform(const torch::Tensor& transform) { 
        AT_ASSERT(transform.sizes() == torch::IntArrayRef({4, 4}), "Transform must be a 4x4 matrix");
        m_transform = transform.clone();
        if (m_transform.scalar_type() != torch::kFloat32) {
            m_transform = m_transform.to(torch::kFloat32);
        }
    }
    
    // Set transforms from position, rotation, and scale vectors
    void setTransform(const std::vector<float>& position, 
                     const std::vector<float>& rotation, 
                     const std::vector<float>& scale) {
        // Ensure valid input
        if (position.size() != 3 || rotation.size() != 3 || scale.size() != 3) {
            throw std::runtime_error("Position, rotation, and scale must be 3-element vectors");
        }
        
        // Create rotation matrices for each axis
        float cosX = cos(rotation[0]);
        float sinX = sin(rotation[0]);
        float cosY = cos(rotation[1]);
        float sinY = sin(rotation[1]);
        float cosZ = cos(rotation[2]);
        float sinZ = sin(rotation[2]);
        
        // Rotation matrix around X-axis
        auto rotX = torch::tensor({
            {1.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, cosX, -sinX, 0.0f},
            {0.0f, sinX, cosX, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        });
        
        // Rotation matrix around Y-axis
        auto rotY = torch::tensor({
            {cosY, 0.0f, sinY, 0.0f},
            {0.0f, 1.0f, 0.0f, 0.0f},
            {-sinY, 0.0f, cosY, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        });
        
        // Rotation matrix around Z-axis
        auto rotZ = torch::tensor({
            {cosZ, -sinZ, 0.0f, 0.0f},
            {sinZ, cosZ, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        });
        
        // Scale matrix
        auto scaleMatrix = torch::tensor({
            {scale[0], 0.0f, 0.0f, 0.0f},
            {0.0f, scale[1], 0.0f, 0.0f},
            {0.0f, 0.0f, scale[2], 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}
        });
        
        // Translation matrix
        auto translationMatrix = torch::tensor({
            {1.0f, 0.0f, 0.0f, position[0]},
            {0.0f, 1.0f, 0.0f, position[1]},
            {0.0f, 0.0f, 1.0f, position[2]},
            {0.0f, 0.0f, 0.0f, 1.0f}
        });
        
        // Combine the transforms: translation * rotation * scale
        // The order is important - we apply scale first, then rotations, then translation
        m_transform = torch::matmul(translationMatrix, 
                      torch::matmul(rotZ, 
                      torch::matmul(rotY, 
                      torch::matmul(rotX, scaleMatrix))));
    }
    
    // Reset transform to identity
    void resetTransformMatrix() {
        m_transform = torch::eye(4);
    }
    
    // Visibility
    bool isVisible() const { return m_visible; }
    void setVisible(bool visible) { m_visible = visible; }
    
    // Material override
    void setMaterial(MaterialHandle materialHandle) { m_materialHandle = materialHandle; }
    
private:
    ResourceManager* m_resourceManager;  // Non-owning pointer to the resource manager
    GeometryHandle m_geometryHandle;     // Handle to the geometry resource
    MaterialHandle m_materialHandle;     // Handle to the material resource
    torch::Tensor m_transform;           // 4x4 transformation matrix
    bool m_visible;                      // Visibility flag
};

} // namespace optix_renderer