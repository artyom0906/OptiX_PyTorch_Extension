from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys
import subprocess
import glob
import shutil

# Path to OptiX SDK
optix_include = os.environ.get('OPTIX_INCLUDE', os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64/include')))

# Path to CUDA
cuda_path = os.environ.get('CUDA_PATH', '/opt/cuda')
cuda_include = os.path.join(cuda_path, 'include')
cuda_lib64 = os.path.join(cuda_path, 'lib64')

# Path to OpenVR
openvr_root = os.environ.get('OPENVR_ROOT', '/home/artem/CLionProjects/openvr-tests/openvr')
openvr_include = os.environ.get('OPENVR_INCLUDE', os.path.join(openvr_root, 'headers'))
openvr_lib = os.environ.get('OPENVR_LIB', os.path.join(openvr_root, 'bin'))

# Path to OpenGL-related libraries (GLEW, GLFW)
glew_include = os.environ.get('GLEW_INCLUDE', '/usr/include/GL')
glfw_include = os.environ.get('GLFW_INCLUDE', '/usr/include/GLFW')

# NVCC path
nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc')

# Compile OptiX kernels
def compile_ptx():
    """Compile all OptiX kernels to PTX files"""
    print("Compiling OptiX kernels to PTX...")

    # Create ptx output directory if it doesn't exist
    os.makedirs('ptx', exist_ok=True)

    # Find all kernel files
    kernel_files = glob.glob('src/kernels/*.cu')
    if not kernel_files:
        print("Warning: No kernel files found in src/kernels/")
        return

    # Ensure our material_types.h file is available
    material_types_file = 'src/kernels/material_types.h'
    if not os.path.exists(material_types_file):
        print(f"Error: {material_types_file} not found")
        sys.exit(1)

    # Compile only specific kernel files - skip legacy_shader.cu due to errors
    for kernel_file in kernel_files:
        output_name = os.path.splitext(os.path.basename(kernel_file))[0]

        # Skip legacy_shader.cu which has compilation errors
        if output_name == 'legacy_shader':
            print(f"Skipping compilation of {kernel_file} - known to have errors")
            continue

        # Make sure we compile our new material_texture.cu shader
        if output_name == 'material_texture':
            print(f"Compiling material system shader: {kernel_file}")

        output_file = f'ptx/{output_name}.ptx'

        cmd = [
            nvcc_path,
            '--compiler-bindir', '/usr/bin/gcc-14',
            '-ptx',
            '-lineinfo',     # Add line information for debugging
            '--use_fast_math',
            f'-I{optix_include}',
            f'-I{cuda_include}',
            f'-I{openvr_include}',  # Add OpenVR include
            f'-I{glew_include}',    # Add GLEW include
            f'-I{glfw_include}',    # Add GLFW include
            f'-I{os.path.join(os.getcwd(), "src")}',
            f'-I{os.path.join(os.getcwd(), "include")}',
            '-std=c++17',
            '-arch=sm_120',  # Adjust for your GPU architecture
            '-o', output_file,
            kernel_file
        ]

        print(f"Compiling {kernel_file} -> {output_file}")
        try:
            result = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
            if result.stderr:
                print(f"Warnings/Info during compilation: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error compiling {kernel_file}: {e.stderr}")
            sys.exit(1)

    print("OptiX kernel compilation completed successfully")

# Extra compile arguments
extra_compile_args = {
    'cxx': [
        '-O2',
        '-std=c++17',
        '-D_GLIBCXX_USE_CXX11_ABI=1',
        '-fvisibility=hidden'
    ],
    'nvcc': [
        '--compiler-bindir', '/usr/bin/gcc-14',
        '-D_GLIBCXX_USE_CXX11_ABI=1',
        '-O3',
        '--use_fast_math',
        '-arch=sm_120',  # Modify according to your GPU
        f'-I{optix_include}',
        f'-I{cuda_include}',
        f'-I{openvr_include}',  # Add OpenVR include
        f'-I{glew_include}',    # Add GLEW include
        f'-I{glfw_include}',    # Add GLFW include
        '-std=c++17',
        '--generate-code=arch=compute_86,code=sm_86',
        '--extended-lambda',
        '--expt-relaxed-constexpr',
    ]
}
# Add this function to find Eigen include directory
def get_eigen_include():
    """Get Eigen include directory using pkg-config"""
    try:
        eigen_include = subprocess.check_output(['pkg-config', '--cflags', 'eigen3']).decode('utf-8').strip()
        if eigen_include.startswith('-I'):
            return eigen_include[2:]  # Remove -I prefix
    except:
        # Fallback paths to check
        common_paths = [
            '/usr/include/eigen3',
            '/usr/local/include/eigen3',
            '/opt/local/include/eigen3',
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    raise RuntimeError("Could not find Eigen3 include directory. Please install libeigen3-dev")

# Find all source files
def find_sources():
    """Find all source files for the extension"""
    source_files = []

    # Add all .cpp files from src directory
    for cpp_file in glob.glob('src/**/*.cpp', recursive=True):
        source_files.append(cpp_file)

    # Add all .cu files from src directory except in kernels/ subdirectory and optix/
    for cu_file in glob.glob('src/**/*.cu', recursive=True):
        # Skip kernel files which are compiled separately
        if not (cu_file.startswith('src/kernels/') or cu_file.startswith('src/optix/')):
            source_files.append(cu_file)

    return source_files

# Custom build extension class
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # First compile the PTX files
        compile_ptx()

        # Then proceed with the normal build
        super().build_extensions()

        # Get build directory for copying PTX files
        build_dir = os.path.abspath(self.build_temp)
        lib_dir = os.path.dirname(self.get_ext_fullpath('optix_resource_system'))

        # Create ptx directory in the package
        ptx_dir = os.path.join(lib_dir, 'ptx')
        os.makedirs(ptx_dir, exist_ok=True)

        # Copy compiled PTX files
        for ptx_file in glob.glob('ptx/*.ptx'):
            dest_path = os.path.join(ptx_dir, os.path.basename(ptx_file))
            shutil.copy2(ptx_file, dest_path)
            print(f"Copied {ptx_file} to {dest_path}")

        # Generate compilation database for CLion
        # First need to find compile_commands.json in the build directory
        compile_commands_path = None
        for root, dirs, files in os.walk(build_dir):
            if 'compile_commands.json' in files:
                compile_commands_path = os.path.join(root, 'compile_commands.json')
                break

        if compile_commands_path:
            # Copy to project root for CLion
            shutil.copy2(compile_commands_path, os.path.join(os.getcwd(), 'compile_commands.json'))
            print(f"Copied compilation database to project root for CLion")

# Get source files
source_files = find_sources()
print(f"Building with the following source files: {source_files}")

# Check if OpenVR is available
try:
    # Try to find openvr_api.h header
    openvr_header = os.path.join(openvr_include, 'openvr.h')
    if not os.path.exists(openvr_header):
        print(f"Warning: OpenVR header not found at {openvr_header}")
        print("OpenVR integration may not work properly.")
    else:
        print(f"Found OpenVR header at {openvr_header}")

    # Try to find libopenvr_api library
    openvr_lib_file = os.path.join(os.path.join(openvr_lib, 'linux64'), 'libopenvr_api.so')
    if not os.path.exists(openvr_lib_file):
        print(f"Warning: OpenVR library not found at {openvr_lib_file}")
        print("Checking alternative locations...")

        # Check common alternative locations
        alt_locations = [
            '/usr/lib/libopenvr_api.so',
            '/usr/local/lib/libopenvr_api.so',
            '/usr/lib/x86_64-linux-gnu/libopenvr_api.so'
        ]

        found = False
        for loc in alt_locations:
            if os.path.exists(loc):
                openvr_lib = os.path.dirname(loc)
                print(f"Found OpenVR library at {loc}")
                found = True
                break

        if not found:
            print("OpenVR library not found in common locations.")
            print("Please set OPENVR_LIB environment variable to the directory containing libopenvr_api.so")
    else:
        print(f"Found OpenVR library at {openvr_lib_file}")

except Exception as e:
    print(f"Error checking OpenVR availability: {e}")

# Display NVCC version
try:
    nvcc_version_output = subprocess.check_output([nvcc_path, '--version']).decode()
    print(f"NVCC version: {nvcc_version_output}")
except Exception as e:
    print(f"Error checking NVCC version: {e}")

# Set up the extension
setup(
    name='optix_resource_system',
    version='0.1.0',
    description='OptiX PyTorch Extension with Resource Management and OpenVR Support',
    author='Artem',
    author_email='example@example.com',
    ext_modules=[
        CUDAExtension(
            name='optix_resource_system',
            sources=source_files,
            include_dirs=[
                optix_include,
                cuda_include,
                openvr_include,
                glew_include,
                glfw_include,
                get_eigen_include(),
                'include',
                'src'
            ],
            library_dirs=[
                cuda_lib64,
                openvr_lib,
                '/usr/lib/x86_64-linux-gnu'
            ],
            libraries=[
                'cudart',
                'cuda',
                'openvr_api',  # OpenVR library
                'GL',          # OpenGL library
                'GLEW',        # GLEW library
                'glfw',        # GLFW library
                'pthread', 'dl'
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    },
    zip_safe=False,
    python_requires='>=3.6',
    package_data={
        'optix_resource_system': ['ptx/*.ptx'],
    }
)