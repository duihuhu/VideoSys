from setuptools import setup
import numpy
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths

import sysconfig

# 添加 PyTorch 的 include 和 library 目录
torch_include_dirs = include_paths()  # 获取 PyTorch 头文件路径
torch_library_dirs = library_paths()  # 获取 PyTorch 库路径
python_include_dir = sysconfig.get_path('include')

# 包含 CUDA 和 C++ 源文件的扩展模块
ext_modules = [
    CUDAExtension(
        name="video_ops",
        sources=["pybind.cpp", "trans_manager.cu", "trans_engine.cu", "trans_worker.cu"],  # 包含 C++ 和 CUDA 文件
        include_dirs=[numpy.get_include(), python_include_dir] + torch_include_dirs,  # 添加 numpy 和 PyTorch 的头文件路径
        library_dirs=torch_library_dirs,  # 添加 PyTorch 的库文件路径
        libraries=['c10', 'torch', 'torch_cpu', 'torch_cuda', 'nccl'],  # 链接 PyTorch 所需的库
        extra_compile_args={
            'cxx': ['-O3'],        # C++ 编译器选项
            'nvcc': ['-O3']        # CUDA 编译器选项
        }
    )
]

# setup 函数
setup(
    name="video_ops",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
