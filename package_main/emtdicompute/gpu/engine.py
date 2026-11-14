"""OptiX closest-hit queries on PyTorch-owned CUDA memory."""
import ctypes as ct
import os
from pathlib import Path
import sys
from threading import RLock

import numpy as np
from platformdirs import user_cache_path
import torch


def cuda_device(device='cuda'):
    requested = torch.device(device)
    if requested.type != 'cuda':
        raise ValueError("emtdicompute-gpu requires a CUDA device; use the CPU distribution for device='cpu'.")
    if not torch.cuda.is_available():
        raise RuntimeError('emtdicompute-gpu requires an NVIDIA GPU, a compatible NVIDIA driver, and CUDA-enabled PyTorch.')
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f'CUDA device index out of range: {index}')
    return torch.device('cuda', index)


class OptixScene:
    """Own the static acceleration structure; callers own all ray/output tensors.

    The instance serializes launches/close on the host. Each launch uses the caller's
    current stream; CUDA events protect the reusable parameter slots.
    """
    def __init__(self, vertices, triangles, device='cuda'):
        self.device = cuda_device(device)
        self._lock = RLock()
        self.handle = ct.c_void_p()
        directory = Path(os.environ.get('EMTDI_GPU_LIBRARY_DIR', Path(__file__).parent / '_native'))
        library = directory / ('emtdi_optix.dll' if sys.platform == 'win32' else 'libemtdi_optix.so')
        ptx = directory / 'rays.ptx'
        if not library.is_file() or not ptx.is_file():
            raise RuntimeError('GPU native binaries are missing. Install a platform wheel or build with '
                               '`python -m emtdicompute.gpu.build --optix-include <OptiX/include>`. '
                               'See README.md for source installation.')
        try:
            self.dll = ct.CDLL(str(library.resolve()))
        except OSError as error:
            raise RuntimeError(f'Cannot load {library}. Check the NVIDIA driver and native runtime dependencies: {error}') from error
        self.dll.gpu_last_error.restype = ct.c_char_p
        self.dll.gpu_abi_version.restype = ct.c_int
        if self.dll.gpu_abi_version() != 1:
            raise RuntimeError('GPU binary ABI mismatch; rebuild the native backend for this package.')
        self.dll.gpu_create.argtypes = [ct.c_int, ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_uint,
                                       ct.c_char_p, ct.c_char_p, ct.POINTER(ct.c_void_p)]
        self.dll.gpu_create.restype = ct.c_int
        self.dll.gpu_direct.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_uint64, ct.c_uint,
                                       ct.c_uint64, ct.c_uint64, ct.c_uint64]
        self.dll.gpu_direct.restype = ct.c_int
        self.dll.gpu_destroy.argtypes = [ct.c_void_p]
        self.dll.gpu_destroy.restype = None
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        triangles = np.ascontiguousarray(triangles, dtype=np.uint32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError('Expected mesh vertices/faces with shape (N,3).')
        if not len(triangles) or np.any(triangles >= len(vertices)) or not np.isfinite(vertices).all():
            raise ValueError('Invalid triangle mesh.')
        cache = Path(os.environ.get('EMTDI_GPU_CACHE_DIR', user_cache_path('emtdicompute-gpu'))) / 'optix'
        cache.mkdir(parents=True, exist_ok=True)
        with torch.cuda.device(self.device):
            torch.cuda.init()
            self._check(self.dll.gpu_create(self.device.index, vertices.ctypes.data, len(vertices),
                triangles.ctypes.data, len(triangles), os.fsencode(ptx.resolve()), os.fsencode(cache.resolve()),
                ct.byref(self.handle)))

    def _check(self, result):
        if result:
            message = self.dll.gpu_last_error().decode('utf-8', errors='replace')
            raise RuntimeError(f'OptiX backend: {message}')

    def trace(self, origin, directions):
        with self._lock, torch.cuda.device(self.device):
            if not self.handle:
                raise RuntimeError('GPU scene is closed.')
            for value in (origin, directions):
                if value.device != self.device or value.dtype != torch.float32 or not value.is_contiguous():
                    raise ValueError(f'Expected contiguous float32 tensors on {self.device}.')
            if origin.shape != (3,) or directions.ndim != 2 or directions.shape[1] != 3:
                raise ValueError('Expected one origin (3,) and directions (N,3).')
            if len(directions) >= 2**30:
                raise ValueError('Too many rays in one launch; split into smaller chunks.')
            ids = torch.empty(len(directions), dtype=torch.int32, device=self.device)
            distances = torch.empty(len(directions), dtype=torch.float32, device=self.device)
            stream = torch.cuda.current_stream(self.device)
            for value in (origin, directions, ids, distances):
                value.record_stream(stream)
            self._check(self.dll.gpu_direct(self.handle, origin.data_ptr(), directions.data_ptr(), len(directions),
                                           ids.data_ptr(), distances.data_ptr(), stream.cuda_stream))
            return ids, distances

    def close(self):
        with self._lock:
            if self.handle:
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize(self.device)
                    self.dll.gpu_destroy(self.handle)
                self.handle = ct.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Python/CUDA may already be shutting down.
