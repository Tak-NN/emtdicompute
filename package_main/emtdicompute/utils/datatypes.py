from dataclasses import dataclass
import numpy as np
import trimesh
import torch

@dataclass
class PyramData:
    pyram_points_allframe: np.ndarray
    lines: np.ndarray
    pyram_side_normals: np.ndarray

@dataclass
class SampleMesh:
    sampled_points: np.ndarray
    face_vh_labels: np.ndarray
    face_normals: np.ndarray
    mesh: trimesh.Geometry
    density: int

@dataclass
class CameraPosDir:
    cam_orig: np.ndarray
    cam_forward: np.ndarray

@dataclass
class Em3diPrelims:
    visible_points_allframes: torch.tensor
    vhlbls_visipts_allframes: torch.tensor
    normals_visipts_allframes: torch.tensor
    frame_indices: torch.tensor
    total_frames: int
