from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import numpy
import torch
from torch.utils.cpp_extension import include_paths, library_paths
import sysconfig
import subprocess

# Custom CUDAExtension class
class CUDAExtension(Extension):
    def __init__(self, name, sources, include_dirs=None, library_dirs=None, libraries=None, extra_compile_args=None):
        super().__init__(name, sources)
        self.include_dirs = include_dirs or []
        self.library_dirs = library_dirs or []
        self.libraries = libraries or []
        self.extra_compile_args = extra_compile_args or {}

# Custom build_ext class to handle nvcc compiler
class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            if any(source.endswith(".cu") for source in ext.sources):
                self._compile_cuda(ext)
            else:
                self._compile_cpp(ext)

    def _compile_cuda(self, ext):
        for source in ext.sources:
            if source.endswith(".cu"):
                output_file = self.get_ext_fullpath(ext.name)
                output_dir = os.path.dirname(output_file)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # nvcc compilation command
                nvcc_cmd = [
                    'nvcc', '-std=c++17', '-shared', '-Xcompiler', '-fPIC',
                    '-o', output_file,
                    source
                ]

                # Include paths and other compilation flags
                nvcc_cmd += ['-I' + inc for inc in ext.include_dirs]
                nvcc_cmd += ['-L' + lib for lib in ext.library_dirs]
                nvcc_cmd += ['-l' + lib for lib in ext.libraries]
                nvcc_cmd += ext.extra_compile_args.get('nvcc', [])

                # Execute the nvcc command
                print(f"Running nvcc command: {' '.join(nvcc_cmd)}")
                try:
                    subprocess.run(nvcc_cmd, check=True)
                except subprocess.CalledProcessError:
                    raise RuntimeError(f"nvcc failed to compile {source}")

    def _compile_cpp(self, ext):
        build_ext.build_extensions(self)

# Get PyTorch's include and library paths
torch_include_dirs = include_paths()
torch_library_dirs = library_paths()
python_include_dir = sysconfig.get_path('include')

# Extension modules
ext_modules = [
    CUDAExtension(
        name="video_ops",
        sources=["pybind.cpp", "trans_manager.cu"],
        include_dirs=[numpy.get_include(), python_include_dir] + torch_include_dirs,
        library_dirs=torch_library_dirs,
        libraries=['c10', 'torch', 'torch_cpu', 'torch_cuda', 'nccl'],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '-lnccl', '-Xcompiler', '-fopenmp']  # Enable OpenMP for parallelism
        }
    )
]

# Setup function
setup(
    name="video_ops",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
