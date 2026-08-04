from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='voxelize',
    ext_moduels=[
        CUDAExtension(
            name='voxelize_cuda',
            sources=[
                'csrc/voxelize_ext.cpp',
                'csrc/voxelize.cu',
            ],
            extra_compile_args={
                'cxx': ['-03'],
                'nvcc': ['-03', '--use_fast_math'],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
