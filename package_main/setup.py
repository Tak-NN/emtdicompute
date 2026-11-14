from setuptools import setup, find_packages
from pathlib import Path

setup(

    name = 'emtdicompute',
    version = '2.0.0',
    description = 'Embodied 3D Isovist metrics with parallel CPU Embree visibility',
    long_description = (Path(__file__).parent/'README.md').read_text(encoding='utf-8'),
    long_description_content_type = 'text/markdown',
    python_requires = '>=3.12',
    packages = find_packages(exclude=['build', 'build.*', 'tests', 'tests.*']),
    install_requires = ['numpy', 'pandas', 'trimesh>=4.9', 'open3d', 'embreex', 'torch'],

)
