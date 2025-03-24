cmake_minimum_required(VERSION 3.18)
project(OptiX_PyTorch_Extension LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

# Find OptiX
set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location")
if(NOT OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR "/home/artem/OptiX_PyTorch_Extension/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64" CACHE PATH "Path to OptiX installed location" FORCE)
endif()
set(OptiX_INCLUDE_DIR "${OptiX_INSTALL_DIR}/include")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OptiX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Add all source files
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.cu"
    "src/*.cuh"
    "include/*.h"
    "include/*.cuh"
)

# This is to generate a compilation database for CLion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86 --extended-lambda --expt-relaxed-constexpr")

# Create a static library target (not for installation, just for IDE integration)
add_library(optix_resource_system STATIC ${SOURCES})
target_link_libraries(optix_resource_system ${TORCH_LIBRARIES} cuda cudart)

# Add kernels subdirectory for PTX compilation
add_subdirectory(src/kernels)

# Set source groups for better IDE organization
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

# Add installation target (this doesn't actually install anything, it's for IDE purposes)
install(TARGETS optix_resource_system DESTINATION lib)

# Add a CMake option to enable/disable tests
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    # If we had C++ tests, we would set them up here
endif()

# Custom target to run Python tests
add_custom_target(python_tests
    COMMAND ${CMAKE_COMMAND} -E echo "Running Python tests..."
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_all_tests.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)