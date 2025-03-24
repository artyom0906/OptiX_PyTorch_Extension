#pragma once

#include <torch/extension.h>
#include <optional>
#include "Resource.h"

namespace optix_renderer {

// Holds geometry data (vertices, indices, etc.)
class GeometryResource : public Resource {
public:
    GeometryResource(ResourceID id, const torch::Tensor& vertices, std::optional<torch::Tensor> indices = std::nullopt)
        : Resource(id, ResourceType::GEOMETRY)
        , m_vertices(vertices.clone()) // Clone to ensure we own the data
        , m_indices(indices.has_value() ? std::optional<torch::Tensor>(indices->clone()) : std::nullopt)
        , m_normals(std::nullopt)
        , m_texCoords(std::nullopt)
        , m_tangents(std::nullopt)
        , m_bitangents(std::nullopt)
    {
        // Ensure vertices are in the right format (float32)
        if (m_vertices.scalar_type() != torch::kFloat32) {
            m_vertices = m_vertices.to(torch::kFloat32);
        }
        
        // Ensure indices are in the right format (int32)
        if (m_indices.has_value() && m_indices->scalar_type() != torch::kInt32) {
            m_indices = m_indices->to(torch::kInt32);
        }
    }
    
    // Data access
    const torch::Tensor& getVertices() const { return m_vertices; }
    const std::optional<torch::Tensor>& getIndices() const { return m_indices; }
    const std::optional<torch::Tensor>& getNormals() const { return m_normals;}
    const std::optional<torch::Tensor>& getTexCoords() const {return m_texCoords;}
    const std::optional<torch::Tensor>& getTangents() const {return m_tangents;}
    const std::optional<torch::Tensor>& getBitangents() const {return m_bitangents;}
    
    // Setters for additional data
    void setNormals(const torch::Tensor& normals) { m_normals = normals.clone(); }
    void setTexCoords(const torch::Tensor& texCoords) { m_texCoords = texCoords.clone(); }
    void setTangents(const torch::Tensor& tangents) { m_tangents = tangents.clone(); }
    void setBitangents(const torch::Tensor& bitangents) { m_bitangents = bitangents.clone(); }
    
    // Geometry operations
    void computeNormals();
    void computeTangentSpace();
    void transform(const torch::Tensor& matrix);

private:
    torch::Tensor m_vertices;                       // [N, 3] float tensor of vertex positions
    std::optional<torch::Tensor> m_indices;         // [M] int tensor of triangle indices (optional)
    std::optional<torch::Tensor> m_normals;         // [N, 3] float tensor of vertex normals (optional)
    std::optional<torch::Tensor> m_texCoords;       // [N, 2] float tensor of texture coordinates (optional)
    std::optional<torch::Tensor> m_tangents;        // [N, 3] float tensor of tangent vectors (optional)
    std::optional<torch::Tensor> m_bitangents;      // [N, 3] float tensor of bitangent vectors (optional)
};

} // namespace optix_renderer