from pathlib import Path
import runpy

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent


class BuildGpu(build_py):
    def run(self):
        super().run()
        build = runpy.run_path(str(ROOT/'emtdicompute/gpu/build.py'))['build']
        build(output_dir=Path(self.build_lib)/'emtdicompute/gpu/_native',
              build_dir=Path(self.get_finalized_command('build').build_temp)/'optix')


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        # The shared library uses ctypes and no CPython/PyTorch C++ ABI.
        return 'py3', 'none', platform


setup(
    name='emtdicompute-gpu',
    version='3.0.0',
    description='Embodied 3D Isovist metrics with CUDA/OptiX visibility',
    long_description=(ROOT/'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    python_requires='>=3.12',
    packages=find_packages(exclude=['build', 'build.*', 'tests', 'tests.*']),
    install_requires=['numpy>=1.26', 'pandas>=2.2', 'trimesh>=4.9', 'open3d>=0.19',
                      'torch>=2.9', 'platformdirs>=4'],
    package_data={'emtdicompute.gpu': ['native/*.cpp', 'native/*.cu', 'native/*.h',
                                      '_native/*.dll', '_native/*.so', '_native/*.ptx', '_native/*.json']},
    cmdclass={'build_py': BuildGpu, 'bdist_wheel': PlatformWheel},
    distclass=BinaryDistribution,
)
