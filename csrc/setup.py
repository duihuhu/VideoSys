from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import os
import numpy
import torch
from torch.utils.cpp_extension import include_paths, library_paths
import sysconfig

# 自定义 CUDAExtension 类
class CUDAExtension(Extension):
    def __init__(self, name, sources, include_dirs=None, library_dirs=None, libraries=None, extra_compile_args=None):
        super().__init__(name, sources)
        self.include_dirs = include_dirs or []
        self.library_dirs = library_dirs or []
        self.libraries = libraries or []
        self.extra_compile_args = extra_compile_args or []

# 自定义 build_ext 来处理 nvcc 编译器
class BuildExt(build_ext):
    def build_extensions(self):
        # 设置编译器
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

                # nvcc 编译命令
                nvcc_cmd = [
                    'nvcc', '-std=c++14', '-shared', '-Xcompiler', '-fPIC',
                    '-o', output_file,
                    source
                ]

                # 处理 include 路径和其他编译选项
                nvcc_cmd += ['-I' + inc for inc in ext.include_dirs]
                nvcc_cmd += ext.extra_compile_args.get('nvcc', [])  # 只传递 nvcc 编译器的选项

                # 执行编译命令
                print(f"Running nvcc command: {' '.join(nvcc_cmd)}")
                if os.system(' '.join(nvcc_cmd)) != 0:
                    raise RuntimeError(f"nvcc failed to compile {source}")

    def _compile_cpp(self, ext):
        build_ext.build_extensions(self)

# 添加 PyTorch 的 include 和 library 目录
torch_include_dirs = include_paths()  # 获取 PyTorch 头文件路径
torch_library_dirs = library_paths()  # 获取 PyTorch 库路径
python_include_dir = sysconfig.get_path('include')

# 包含 CUDA 和 C++ 源文件的扩展模块
ext_modules = [
    CUDAExtension(
        name="video_ops",
        sources=["pybind.cpp", "trans_manager.cu"],  # 包含 C++ 和 CUDA 文件
        include_dirs=[numpy.get_include(), python_include_dir] + torch_include_dirs,  # 添加 numpy 和 PyTorch 的头文件路径
        library_dirs=torch_library_dirs,  # 添加 PyTorch 的库文件路径
        libraries=['c10', 'torch', 'torch_cpu', 'torch_cuda', 'nccl'],  # 链接 PyTorch 所需的库
        extra_compile_args={
            'cxx': ['-O3'],        # C++ 编译器选项
            'nvcc': ['-O3', "-lnccl"]        # CUDA 编译器选项
        }
    )
]

# setup 函数
setup(
    name="video_ops",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
