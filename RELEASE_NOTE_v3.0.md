# emtdicompute 3.0.0-gpu

The GPU edition is released as **emtdicompute 3.0.0-gpu**, separately from the
**emtdicompute 2.0.0** CPU-prallel edition. Both use `import emtdicompute`;
install them into separate virtual environments.

## Features

- CUDA PyTorch FOV filtering, ray preparation, visibility classification,
  visible-point extraction, and metric calculation connected on GPU memory.
- OptiX closest-hit intersection on two-sided triangles; verified source-face
  IDs preserve self-face visibility and the existing distance tolerance.
- Reusable `GpuSession` holds the static point cloud and acceleration structure
  across chunks. The batch API creates and reuses a session automatically.
- PyTorch and OptiX share GPU buffers on the caller's CUDA stream.
- `run_batch_job(force_usecache=True)` forwards explicit cache selection to the
  sampler; GPU visibility still requires verified source triangle IDs.
- The validated direction normalization, visibility rules, and metric formulas
  are retained. Existing float32 accumulation variability remains.

## Requirements and installation

Requires Python 3.12+, CUDA-enabled PyTorch 2.9+, and a compatible NVIDIA GPU and
driver. The tested environment is Windows x64, RTX 5070, driver 610.88,
PyTorch 2.9.1+cu130. CUDA Toolkit 12.8, OptiX 9.1 headers, and MSVC were used to
build the Windows wheel. Users installing that wheel need the driver and runtime
dependencies, but do not need a local CUDA Toolkit, OptiX SDK, or compiler.

After installing the appropriate CUDA-enabled PyTorch, run from the repository root:

```powershell
python -m pip install --upgrade .\package_release\emtdicompute_gpu-3.0.0-py3-none-win_amd64.whl
python -c "import emtdicompute; print(emtdicompute.__version__)"
```

The package version should report `3.0.0`. See [the installation/API guide](package_main/README.md)
for source builds, device/session usage, and cache behavior. A source archive is
also provided in `package_release/`.

This edition requires CUDA and has no CPU fallback. Linux source-build support
and multiple CUDA device selection are implemented, but Linux and multi-GPU
hardware are not validated here. The existing small triangle-boundary differences
from CPU Embree and float32 aggregation variability mean CSV results are not
guaranteed to be bit-identical to the CPU edition or between CUDA runs.
