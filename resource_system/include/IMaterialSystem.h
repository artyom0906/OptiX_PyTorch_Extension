#pragma once

// Include the core resource types which already contains the basic IMaterialSystem
#include "ResourceTypes.h"

#include <string>
#include <unordered_map>
#include <any>
#include <vector>
#include <optix.h>

namespace optix_renderer {

/**
 * Enhanced interface for material systems that can be integrated with the resource system
 * 
 * This interface extends the basic IMaterialSystem with additional MDL capabilities,
 * allowing different implementations (like MDL, OptiX built-in, custom) to be used interchangeably.
 * 
 * MDL integration guide:
 * 1. Implement this interface for MDL support
 * 2. Connect to NVIDIA MDL SDK
 * 3. Compile MDL materials to OptiX compatible programs
 * 4. Manage material parameters and textures
 * 5. Generate device-specific representations
 */
class IMDLMaterialSystem : public IMaterialSystem {
public:
    virtual ~IMDLMaterialSystem() = default;
    
    /**
     * Initializes the material system
     * 
     * For MDL integration:
     * - Initialize the MDL SDK
     * - Set up the MDL search paths
     * - Configure the MDL compiler
     */
    virtual bool initialize() = 0;
    
    /**
     * Loads a material definition from a file (MDL specific)
     *
     * @param filename The MDL file to load
     * @param materialName The name of the material in the file
     * @return A handle to the material description
     */
    virtual MaterialHandle loadMaterialFromFile(const std::string& filename, 
                                              const std::string& materialName) = 0;
    
    /**
     * Creates a material instance with default parameters
     *
     * @param materialDesc The material description handle
     * @return A handle to the created material instance
     */
    virtual MaterialHandle createMaterialInstance(MaterialHandle materialDesc) = 0;
    
    /**
     * Sets a parameter value on a material using std::any
     *
     * @param material The material instance handle
     * @param paramName The name of the parameter
     * @param value The parameter value as std::any
     * @return True if successful
     */
    virtual bool setParameter(MaterialHandle material, 
                            const std::string& paramName, 
                            std::any value) = 0;
    
    /**
     * Sets a texture parameter on a material
     *
     * @param material The material instance handle
     * @param paramName The name of the texture parameter
     * @param textureHandle Handle to the texture resource
     * @return True if successful
     */
    virtual bool setTextureParameter(MaterialHandle material,
                                   const std::string& paramName,
                                   TextureHandle textureHandle) = 0;
    
    /**
     * Compiles the material for a specific device with OptiX context
     * This overload handles device-specific compilation with OptixDeviceContext
     *
     * @param material The material instance handle
     * @param deviceContext The OptiX device context
     * @param compileOptions Compilation options
     * @return True if successful
     */
    virtual bool compileMaterial(MaterialHandle material,
                               OptixDeviceContext deviceContext,
                               void* compileOptions) = 0;
    
    /**
     * Gets the compiled program group for a material
     *
     * @param material The material instance handle
     * @param deviceContext The OptiX device context
     * @return The OptiX program group for the material
     */
    virtual OptixProgramGroup getProgramGroup(MaterialHandle material,
                                           OptixDeviceContext deviceContext) = 0;
    
    /**
     * Gets the parameter data needed for rendering
     *
     * @param material The material instance handle
     * @param deviceContext The OptiX device context
     * @return The parameter data
     */
    virtual void* getParameterData(MaterialHandle material,
                                OptixDeviceContext deviceContext) = 0;
    
    /**
     * Gets the size of the parameter data
     *
     * @param material The material instance handle
     * @param deviceContext The OptiX device context
     * @return The size of the parameter data in bytes
     */
    virtual size_t getParameterDataSize(MaterialHandle material,
                                     OptixDeviceContext deviceContext) = 0;
};

} // namespace optix_renderer