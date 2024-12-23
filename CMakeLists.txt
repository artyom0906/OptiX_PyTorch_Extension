cmake_minimum_required(VERSION 3.26)
project(OptixPyTorchExtension)
enable_language(CUDA)
find_package(CUDAToolkit 12.6 REQUIRED)
find_package(PythonLibs REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(Torch_DIR "E:/code/cuda/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH ${Torch_DIR})

find_package(Torch REQUIRED)

set(SOURCE_FILES old/renderer.cu)
set(Torch_DIR libtorch/share/cmake/Torch)

add_executable( OptixPyTorchExtension ${SOURCE_FILES} E:/code/cuda/libtorch/lib)
target_include_directories(OptixPyTorchExtension PUBLIC ${CUDAToolkit_INCLUDE_DIRS} CUDA::cudart)
target_link_libraries( OptixPyTorchExtension “${TORCH_LIBRARIES}”)

set(OPTIX_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/include")
message("OptixDir: ${OPTIX_DIR}")

target_include_directories(OptixPyTorchExtension PUBLIC ${OPTIX_DIR})
include_directories(${PYTHON_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
set_target_properties(OptixPyTorchExtension PROPERTIES CUDA_ARCHITECTURES "86")