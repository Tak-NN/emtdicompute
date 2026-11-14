# emtdicompute GPU

This `v3.0.0-gpu-acceleration` branch builds **emtdicompute 3.0.0-gpu**, the CUDA/OptiX edition of the Embodied 3D Isovist package. The Python import name remains `emtdicompute`. The CPU-parallel edition is released separately as `emtdicompute 2.0.0`.

It evaluates a limited field of view in a three-dimensional building, supporting point, path, and field analyses. FOV filtering, ray preparation, closest-hit intersection, visible-point selection, and metric calculation run on the GPU. File loading, mesh sampling, camera setup, orchestration, and CSV output remain on the CPU.

The article introducing the framework is available at [Full-text](https://www.researchgate.net/publication/416018981_A_dynamic_and_viewing-direction-aware_3D_visuospatial_analysis_framework_for_indoor_environments_based_on_embodied_3D_isovist) / [Publisher](https://doi.org/10.1177/23998083261496137).

# Installation and use

See [the GPU installation and API guide](package_main/README.md) for platform requirements, wheel/source installation, native builds, and examples.

- Distribution: `emtdicompute-gpu`; import: `emtdicompute`.
- Requires CUDA-enabled PyTorch and an NVIDIA GPU/driver supporting the OptiX backend.
- Tested on Windows x64, Python 3.12, RTX 5070, driver 610.88, PyTorch 2.9.1+cu130, CUDA Toolkit 12.8, and OptiX 9.1 headers.
- This GPU edition has no CPU visibility fallback and does not require Embree/embreex.
- Use separate environments for the CPU and GPU distributions, which share the import namespace.

# Visibility computation

A reusable `GpuSession` keeps the point cloud, normals, labels, source triangle IDs, and OptiX acceleration structure on the selected CUDA device. Batch calls reuse one session across chunks. OptiX accesses PyTorch-owned GPU memory on the caller's current CUDA stream. Large ray and intersection arrays do not pass through the CPU.

Queries return the closest triangle ID and distance, with two-sided triangles. A hit on the sampled point's source triangle is visible; other hits are compared with the point distance using the existing `1e-4` tolerance. Adjacent coplanar triangles remain separate. The validated two-stage direction normalization and scene transformation are retained. The metric definitions and existing float32 PyTorch aggregations are unchanged.

`SampleMesh.face_indices` is required for GPU visibility. `mesh_sampling()` supplies it for new point clouds and verified caches. Legacy caches are reused only when source triangle IDs can be verified without changing their points. Regenerate an unverified cache with `force_recompute=True`; this edition does not silently use the old CPU visibility algorithm.

# Sample codes and reproduction

Example scripts are in `sample_codes/`. The code and data used in the original article are in `reproduction/`. Earlier scripts that explicitly select `device='cpu'` need to select a CUDA device for this edition. Minor triangle-boundary differences from the CPU engine and float32 GPU aggregation variability mean results need not be bit-identical to the original implementation.

# Updates

- v3.0.0: GPU release with versioned wheel/source distributions, persistent GPU sessions, and the batch cache-selection option. See [release notes](RELEASE_NOTES_v3.0.md).
