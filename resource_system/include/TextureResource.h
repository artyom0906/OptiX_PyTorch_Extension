#pragma once

#include <torch/extension.h>
#include "Resource.h"

namespace optix_renderer {

// Holds texture data for materials
class TextureResource : public Resource {
public:
    TextureResource(ResourceID id, const torch::Tensor& data, TextureType type = TextureType::RGB)
        : Resource(id, ResourceType::TEXTURE)
        , m_data(data.clone().contiguous()) // Clone to ensure we own the data
        , m_textureType(type)
        , m_channels(0)
        , m_originalChannels(0)
        , m_width(0)
        , m_height(0)
        , m_originalDataType(data.scalar_type())
    {
        // Validate and determine texture dimensions
        AT_ASSERT(data.dim() >= 2 && data.dim() <= 4, "Texture data must be 2D, 3D, or 4D tensor");
        
        if (data.dim() == 2) {
            // [H, W] grayscale
            m_height = data.size(0);
            m_width = data.size(1);
            m_channels = 1;
            m_originalChannels = 1;
        } else if (data.dim() == 3) {
            // [H, W, C] or [C, H, W]
            if (data.size(2) <= 4) {
                // [H, W, C]
                m_height = data.size(0);
                m_width = data.size(1);
                m_channels = data.size(2);
                
                // For RGB textures (3 channels), convert to RGBA (4 channels) for CUDA compatibility
                if (m_channels == 3) {
                    // Create alpha channel
                    torch::Tensor alpha = torch::ones({m_height, m_width, 1}, m_data.options());
                    // Concatenate RGB with alpha to make RGBA
                    torch::Tensor rgba = torch::cat({m_data, alpha}, 2);
                    m_data = rgba;
                    // Keep track of original channel count for proper handling
                    m_originalChannels = 3;
                    m_channels = 4;
                } else {
                    m_originalChannels = m_channels;
                }
            } else {
                // [C, H, W]
                int originalChannels = data.size(0);
                m_height = data.size(1);
                m_width = data.size(2);
                
                // Convert to [H, W, C] layout for consistency
                m_data = m_data.permute({1, 2, 0}).contiguous();
                m_channels = originalChannels;
                
                // For RGB textures (3 channels), convert to RGBA (4 channels) for CUDA compatibility
                if (m_channels == 3) {
                    // Create alpha channel
                    torch::Tensor alpha = torch::ones({m_height, m_width, 1}, m_data.options());
                    // Concatenate RGB with alpha to make RGBA
                    torch::Tensor rgba = torch::cat({m_data, alpha}, 2);
                    m_data = rgba;
                    // Keep track of original channel count for proper handling
                    m_originalChannels = 3;
                    m_channels = 4;
                } else {
                    m_originalChannels = m_channels;
                }
            }
        } else if (data.dim() == 4) {
            // [B, C, H, W] - Use first batch
            int originalChannels = data.size(1);
            m_height = data.size(2);
            m_width = data.size(3);
            
            // Convert to [H, W, C] layout for consistency
            m_data = m_data[0].permute({1, 2, 0}).contiguous();
            m_channels = originalChannels;
            
            // For RGB textures (3 channels), convert to RGBA (4 channels) for CUDA compatibility
            if (m_channels == 3) {
                // Create alpha channel
                torch::Tensor alpha = torch::ones({m_height, m_width, 1}, m_data.options());
                // Concatenate RGB with alpha to make RGBA
                torch::Tensor rgba = torch::cat({m_data, alpha}, 2);
                m_data = rgba;
                // Keep track of original channel count for proper handling
                m_originalChannels = 3;
                m_channels = 4;
            } else {
                m_originalChannels = m_channels;
            }
        }
        
        // Remember original data type before conversion
        m_originalDataType = m_data.scalar_type();
        
        // Convert to appropriate format based on input type
        if (m_originalDataType == torch::kHalf) {
            // Handle fp16 (half precision) textures
            // For CUDA, we'll still convert to float32, but we'll remember it was half precision
            m_data = m_data.to(torch::kFloat32);
            m_isHalfPrecision = true;
        } else if (m_originalDataType == torch::kUInt8 || m_originalDataType == torch::kInt8) {
            // Handle int8/uint8 textures
            // For CUDA, we'll convert to float32 in [0,1] range
            m_data = m_data.to(torch::kFloat32);
            if (m_originalDataType == torch::kUInt8) {
                m_data = m_data / 255.0f;  // Normalize uint8 values to [0,1]
            } else {
                m_data = (m_data + 128.0f) / 255.0f;  // Normalize int8 values to [0,1]
            }
            m_isInt8Precision = true;
        } else if (m_data.scalar_type() != torch::kFloat32) {
            // Convert any other format to float32
            m_data = m_data.to(torch::kFloat32);
        }
        
        // Clamp values to [0,1] range for regular textures
        if (type != TextureType::HDR) {
            m_data = torch::clamp(m_data, 0.0f, 1.0f);
        }
    }
    
    // Data access
    const torch::Tensor& getData() const { return m_data; }
    TextureType getTextureType() const { return m_textureType; }
    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    int getChannels() const { return m_channels; }
    int getOriginalChannels() const { return m_originalChannels; }
    torch::ScalarType getOriginalDataType() const { return m_originalDataType; }
    bool isHalfPrecision() const { return m_isHalfPrecision; }
    bool isInt8Precision() const { return m_isInt8Precision; }
    
    // Texture operations
    void resize(int width, int height);
    void gammaCorrect(float gamma = 2.2f);
    
private:
    torch::Tensor m_data;             // [H, W, C] texture data, always float32 for CUDA compatibility
    TextureType m_textureType;        // Type of texture (RGB, normal map, etc.)
    int m_channels;                   // Number of channels (might be padded to 1, 2, or 4)
    int m_originalChannels;           // Original number of channels before padding
    int m_width;                      // Texture width
    int m_height;                     // Texture height
    torch::ScalarType m_originalDataType; // Original tensor data type before conversion
    bool m_isHalfPrecision = false;   // True if original data was fp16 (half precision)
    bool m_isInt8Precision = false;   // True if original data was uint8/int8
};

} // namespace optix_renderer