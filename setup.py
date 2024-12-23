# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import subprocess

optix_include = 'C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/include'  # Your OptiX include path
#optix_include = '/mnt/c/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/include'  # Your OptiX include path
#, 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'
cuda_path = os.environ.get('CUDA_PATH')
cuda_include = os.path.join(cuda_path, 'include')
cuda_lib64 = os.path.join(cuda_path, 'lib', 'x64')

extra_compile_args = {
    'cxx': ['/O2', '/MD', '/D_CRT_SECURE_NO_WARNINGS'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-arch=sm_86',  # Modify according to your GPU
        '-I{}'.format(optix_include),
        '-I{}'.format(cuda_include),
        #'-std=c++14',
    ]
}

# Get the correct nvcc path
nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc.exe')

# Verify nvcc version
nvcc_version_output = subprocess.check_output([nvcc_path, '--version']).decode()
print("nvcc version:", nvcc_version_output)

# Pass the nvcc executable path to the build extension
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        self.compiler.set_executable('compiler_so', nvcc_path)
        super().build_extensions()

        # Generate compile_commands.json using Ninja
        build_dir = os.path.abspath(self.build_temp)  # Build directory where `build.ninja` is generated
        if os.path.exists(os.path.join(build_dir, 'build.ninja')):
            try:
                subprocess.run(
                    ['ninja', '-t', 'compdb'],
                    cwd=build_dir,
                    stdout=open(os.path.join(build_dir, 'compile_commands.json'), 'w'),
                    check=True
                )
                print("Generated compile_commands.json in build directory.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate compile_commands.json: {e}")

setup(
    name='optix_renderer',
    ext_modules=[
        CUDAExtension(
            name='optix_renderer',
            sources=['src/OptixRenderer.cpp', 'src/vertices.cu', 'src/TextureObject.cpp'],
            include_dirs=[optix_include, cuda_include],
            library_dirs=[cuda_lib64],  # Only CUDA libraries
            libraries=['cudart', 'Advapi32'],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)
