#pragma once

#include <string>
#include <unordered_map>
#include <any>
#include <vector>
#include <optix.h>

#include "../include/ResourceTypes.h"

namespace optix_renderer {

/**
 * Interface for material systems that can be integrated with the resource system
 * 
 * This interface defines the contract for material systems, allowing different
 * implementations (like MDL, OptiX built-in, custom) to be used interchangeably.
 * 
 * MDL integration guide:
 * 1. Implement this interface for MDL support
 * 2. Connect to NVIDIA MDL SDK
 * 3. Compile MDL materials to OptiX compatible programs
 * 4. Manage material parameters and textures
 * 5. Generate device-specific representations
 */
class IMaterialSystem {
public:
    virtual ~IMaterialSystem() = default;
    
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
     * Loads a material definition from a file
     * 
     * For MDL integration:
     * - Load the MDL module
     * - Retrieve the material definition
     * - Create a material instance template
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
     * For MDL integration:
     * - Instantiate the material with default parameters
     * - Create callable programs for the material functions
     * 
     * @param materialDesc The material description handle
     * @return A handle to the created material instance
     */
    virtual MaterialHandle createMaterialInstance(MaterialHandle materialDesc) = 0;
    
    /**
     * Sets a parameter value on a material
     * 
     * For MDL integration:
     * - Update the material instance parameter
     * - Mark the material for recompilation if needed
     * 
     * @param material The material instance handle
     * @param paramName The name of the parameter
     * @param value The parameter value
     * @return True if successful
     */
    virtual bool setParameter(MaterialHandle material, 
                            const std::string& paramName, 
                            std::any value) = 0;
    
    /**
     * Sets a texture parameter on a material
     * 
     * For MDL integration:
     * - Update the material instance texture parameter
     * - Link to the texture resource
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
     * Compiles the material for a specific device
     * 
     * For MDL integration:
     * - Compile the material for the target device
     * - Generate OptiX compatible programs
     * - Set up the material parameters for the device
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
     * For MDL integration:
     * - Retrieve the compiled program group
     * - Ensure it's valid for the current device
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
     * For MDL integration:
     * - Prepare the parameter data for the shader
     * - Include all necessary textures, values, etc.
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