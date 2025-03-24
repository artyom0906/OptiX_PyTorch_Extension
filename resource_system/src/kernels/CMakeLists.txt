# CMakeLists.txt for compiling OptiX kernels
cmake_minimum_required(VERSION 3.18)

# Check if we have CUDA and OptiX
find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)

set(KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PTX_OUTPUT_DIR ${CMAKE_BINARY_DIR}/ptx)

# Create PTX output directory
file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

# Function to compile a CUDA file to PTX
function(compile_ptx cuda_file output_name)
    get_filename_component(cuda_file_name ${cuda_file} NAME_WE)
    set(output_file "${PTX_OUTPUT_DIR}/${output_name}.ptx")
    
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CUDA_NVCC_EXECUTABLE}
            -ptx
            -I${OptiX_INCLUDE_DIR}
            -I${KERNELS_DIR}
            -I${CMAKE_SOURCE_DIR}/include
            -I${CMAKE_SOURCE_DIR}/src
            -arch=sm_86
            -o ${output_file}
            ${cuda_file}
        DEPENDS ${cuda_file}
        COMMENT "Compiling ${cuda_file} to PTX"
        VERBATIM
    )
    
    add_custom_target(
        ${output_name}_ptx ALL
        DEPENDS ${output_file}
    )
    
    # Add a custom command to copy the PTX file to the Python package
    add_custom_command(
        TARGET ${output_name}_ptx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${output_file}
            ${CMAKE_SOURCE_DIR}/ptx/${output_name}.ptx
        COMMENT "Copying ${output_file} to package directory"
        VERBATIM
    )
endfunction()

# Compile all OptiX kernel files
compile_ptx(${KERNELS_DIR}/shader_minimal.cu "shader_minimal")
compile_ptx(${KERNELS_DIR}/material_texture.cu "material_texture")
compile_ptx(${KERNELS_DIR}/material_texture_simple.cu "material_texture_simple")

# Create a symbolic target for IDEs
add_custom_target(
    kernels
    DEPENDS shader_minimal_ptx material_texture_ptx material_texture_simple_ptx
)