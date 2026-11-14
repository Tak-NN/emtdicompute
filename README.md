# Overview

This branch is the **emtdicompute 2.0.0 CPU-parallel-computing edition**, combining
configurable CPU Embree parallelism with reduced PyTorch data transfers. An NVIDIA
GPU and CUDA Toolkit are **not required**. PyTorch CPU-only builds are
supported. Optional CUDA-enabled PyTorch can accelerate FOV/metric calculations.

*emtdicompute* is a Python package for computing Embodied 3D Isovist metrics in three-dimensional built environments.  
It supports point-, path-, and field-based analyses while explicitly incorporating a limited field of view (FOV).  
The package is designed to facilitate reproducible research workflows in architectural analysis, spatial cognition, and indoor navigation experiments.

An article introducing this package is available here: [Full-text](https://www.researchgate.net/publication/416018981_A_dynamic_and_viewing-direction-aware_3D_visuospatial_analysis_framework_for_indoor_environments_based_on_embodied_3D_isovist) / [Publisher](https://doi.org/10.1177/23998083261496137)

# Installation

## Requirements

- Python 3.12  
- pip  
- (Optional) CUDA-enabled PyTorch and a compatible NVIDIA driver for FOV/metric acceleration

## Dependencies

```bash
pip install trimesh open3d pandas numpy embreex
```

Note: You can run `pip install trimesh[easy]` instead of `pip install trimesh`.

**_In addition, install PyTorch manually_** according to your environment:
- https://pytorch.org/get-started/locally/

For CPU-only use, install the CPU build of PyTorch first, then install this
package. CUDA and OptiX are not required for the CPU execution path.

## Installing from a wheel

Use a virtual environment separate from the GPU edition, which shares the
`emtdicompute` import name. Install your preferred PyTorch build first (the CPU
build for CPU-only use), as described above.

The CPU-edition wheel is provided in `package_release/`. From the repository
root, install it into the Python environment used for your analysis:

```bash
python -m pip install "./package_release/emtdicompute-2.0.0-py3-none-any.whl"
```

For a downloaded wheel, replace the path with its actual location. pip installs
the other declared dependencies automatically. CUDA, OptiX, and a local C++
compiler are not required by this wheel; Embree is provided by the `embreex`
dependency.

When replacing an installed wheel with an updated file carrying the same version
number, force reinstallation. If the dependencies are already installed, use
`--no-deps` to keep them:

```bash
python -m pip install --force-reinstall --no-deps "./package_release/emtdicompute-2.0.0-py3-none-any.whl"
```

Verify the installed version and the module location:

```bash
python -c "import emtdicompute; print(emtdicompute.__version__); print(emtdicompute.__file__)"
```

Select `device='cpu'` in `run_batch_job()` to use only the CPU; `num_workers`
controls the CPU intersection thread limit. Omitting `device` selects CUDA when
available, otherwise CPU. Restart an already-running Python/notebook session
after replacing the installed package.

## Installing the _emtdicompute_ package from this repository
Tha package source is located in `package_main/`.

1. Move to the package directory:

```bash
cd package_main
```

2. Install the package.

For a standard installation:

```bash
pip install .
```

For development (editable install):

```bash
pip install -e .
```

3. Verify the installation:

```bash
python -c "import emtdicompute; print('emtdicompute import: OK')"
```

# Visibility computation

Visibility queries use CPU Embree closest-hit triangle IDs and distances. A hit on
the sampled point's source triangle is treated as visible; hits on other triangles
are compared with the point distance using the existing `1e-4` tolerance. Adjacent
coplanar triangles remain separate. FOV filtering and metric computation still use
PyTorch on the selected device.

Set `num_workers` to control the maximum number of CPU threads used for the
closest-hit queries. The default `1` retains serial execution. The same argument
is accepted by `run_batch_job()`, `compute_em3di_metrics_batch()`,
`compute_em3di_metrics()`, and `embodied_3d_isovist_prelims()`.

```python
from emtdicompute.main import run_batch_job

run_batch_job(
    './path-to-trajectory-records', './path-to-building-obj',
    sampling_density=200,
    num_workers=8,
    device='cpu',
)
```

`num_workers` must be a positive integer. It controls Embree query concurrency,
not PyTorch's own CPU threads or CUDA execution. Small ray groups run serially;
larger groups are split across up to the specified number of workers, with a
thread pool reused across viewpoints in each chunk. Results retain their original
point order. Parallel queries require verified `SampleMesh.face_indices`; legacy
data without verified IDs continue to use the original serial visibility path.

`device='cpu'` runs FOV filtering, ray preparation, visible-point selection, and
metric computation on CPU using PyTorch, with CPU Embree for intersections.
Omit `device` (or use `None`) for automatic CUDA/CPU selection. The same automatic
selection is available in both pipeline APIs and `embodied_3d_isovist_prelims()`.
Explicitly requesting unavailable CUDA raises an error with CPU usage guidance.

With `device='cuda'`, CPU Embree still handles intersections, but point/normal/label
tensors stay on CUDA. Only the visibility mask is sent back for selection; normals,
labels, and selected points no longer make an unnecessary CPU round trip. The same
selection works on CPU, preserving point order, dtype, and frame assignments.
The metric formulas, closest-hit/self-face rules, and `1e-4` tolerance are unchanged.

The separate `emtdicompute-gpu` edition uses OptiX for GPU intersections. Use
separate virtual environments for these distributions because both import as
`emtdicompute`.

`mesh_sampling()` now returns `SampleMesh.face_indices`, and newly generated CSV
caches store this column (cache version 2). Existing caches are loaded without
resampling or rewriting their points: source IDs are recovered only when source
metadata, expected ordering, face normals and point positions on triangles agree.
Unverified caches, including forced caches for a changed source, use the original
visibility calculation. Manually constructed five-argument `SampleMesh` objects
remain supported; provide valid `face_indices` to enable the faster calculation.

# Sample Codes
Example scripts are available in `reproduction/`.

# Reproduction Procedure

To reproduce the analyses presented in the original article, use the code and data provided in `reproduction/`.

# Updates

- v2.0.0: configurable CPU intersection threads, reduced data transfers, automatic
  CPU/CUDA selection, and an explicit CPU-only execution option.
- v1.0 (2026-02-12)
  - Initial public release.
