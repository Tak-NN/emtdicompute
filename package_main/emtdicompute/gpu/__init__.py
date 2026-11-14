"""CUDA/OptiX visibility backend. Importing this module does not initialize CUDA."""
from .session import GpuSession

__all__ = ['GpuSession']
