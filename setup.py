from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='voxelize',
    packages=['voxelize'],
    ext_modules=[
        CUDAExtension(
            name='voxelize_cuda',
            sources=[
                'csrc/voxelize_ext.cpp',
                'csrc/voxelize.cu',
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['--use_fast_math', '--extended-lambda'],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
