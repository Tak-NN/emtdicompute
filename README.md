# Overview

*emtdicompute* is a Python package for computing Embodied 3D Isovist metrics in three-dimensional built environments.
It supports point-, path-, and field-based analyses while incorporating a limited field of view (FOV).
The package is designed for reproducible research workflows in architectural analysis, spatial cognition studies, and indoor navigation experiments.

An article introducing this package is available here: [unpblished].

# Installation Guide

## Requirements
- Python 3.12
- pip
- (Optional) CUDA Toolkit, if you want to use an NVIDIA GPU

## Dependencies
### Install the minimum core dependencies manually

```bash
pip install trimesh open3d pandas numpy embreex rtree scipy
```

Note: You can run `pip install trimesh[easy]` instead of `pip install trimesh`.

In addition, install PyTorch manually based on your environment:
- https://pytorch.org/get-started/locally/

## Installing the **emtdicompute** package from this repository
This repository contains the package source under `package_main/`.

1. Move to the package directory:

```bash
cd package_main
```

2. Install the package.

Editable installation (recommended for development):

```bash
pip install -e .
```

Alternatively, perform a standard installation:

```bash
pip install .
```

3. Verify the installation:

```bash
python -c "import emtdicompute; print('emtdicompute import: OK')"
```

# Reproduction Procedure

To reproduce the analyses presented in the original article, you can use the code and files under `reproduction/`.
The scripts are also helpful as the sample codes.

# Updates

- v1.0 (2026-02-12)
  - Initial public release.



