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
            '-ptx',
            '-lineinfo',     # Add line information for debugging
            '--use_fast_math',
            f'-I{optix_include}',
            f'-I{cuda_include}',
            f'-I{os.path.join(os.getcwd(), "src")}',
            f'-I{os.path.join(os.getcwd(), "include")}',
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
        '-D_GLIBCXX_USE_CXX11_ABI=1',
        '-O3',
        '--use_fast_math',
        '-arch=sm_120',  # Modify according to your GPU
        f'-I{optix_include}',
        f'-I{cuda_include}',
        '-std=c++17',
        '--generate-code=arch=compute_86,code=sm_86',
        '--extended-lambda',
        '--expt-relaxed-constexpr',
    ]
}

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
    description='OptiX PyTorch Extension with Resource Management',
    author='Artem',
    author_email='example@example.com',
    ext_modules=[
        CUDAExtension(
            name='optix_resource_system',
            sources=source_files,
            include_dirs=[optix_include, cuda_include, 'include', 'src'],
            library_dirs=[cuda_lib64, '/usr/lib/x86_64-linux-gnu'],
            libraries=['cudart', 'cuda'],
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