"""Keep static mesh data on CUDA across viewpoints and chunks."""
from dataclasses import replace

import numpy as np
import torch

from emtdicompute.utils.datatypes import Em3diPrelims
from .engine import OptixScene, cuda_device


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class GpuSession:
    """Reusable CUDA data and OptiX scene for one immutable SampleMesh.

    Use as a context manager. Do not mutate the mesh or its sampled arrays during
    its lifetime. The batch API creates/reuses a session automatically.
    """
    def __init__(self, mesh, device='cuda'):
        self.device = cuda_device(device)
        self.mesh = mesh
        self._closed = False
        self.scene = None
        points = _array(mesh.sampled_points)
        normals = _array(mesh.face_normals)
        labels = _array(mesh.face_vh_labels)
        faces = _array(mesh.face_indices) if mesh.face_indices is not None else None
        if faces is None:
            raise ValueError('GPU visibility requires verified face_indices. Regenerate an unverified point-cloud '
                             'cache with mesh_sampling(force_recompute=True), or supply the source triangle ID of each point.')
        if faces.shape != (len(points),) or not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0) or np.any(faces >= len(mesh.mesh.faces)):
            raise ValueError('face_indices must contain one valid source triangle ID per sampled point.')
        if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or labels.shape != (len(points),):
            raise ValueError('Expected points/normals (N,3) and labels (N,).')
        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise ValueError('Point coordinates and normals must be finite.')
        if not len(mesh.mesh.faces) or not np.isfinite(mesh.mesh.vertices).all() or not np.isfinite(mesh.mesh.scale) or mesh.mesh.scale <= 0:
            raise ValueError('Expected a finite triangle mesh with nonzero extent.')
        # Same transformation as the validated CPU/OptiX prototypes, without Embree.
        self.offset = np.asarray(mesh.mesh.vertices, dtype=np.float64).min(axis=0)
        self.scale = 100.0 / mesh.mesh.scale
        vertices = np.ascontiguousarray((mesh.mesh.vertices - self.offset) * self.scale, dtype=np.float32)
        with torch.cuda.device(self.device):
            self.points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            self.labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            self.normals = torch.as_tensor(normals, dtype=torch.float32, device=self.device)
            self.faces = torch.as_tensor(faces, dtype=torch.int32, device=self.device)
            self.offset_gpu = torch.as_tensor(self.offset, dtype=torch.float64, device=self.device)
            self.scene = OptixScene(vertices, mesh.mesh.faces, self.device)
            # Protect construction on one stream followed by use on another.
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(self.device))
        self.sampled_mesh = replace(mesh, sampled_points=self.points)

    def check(self, mesh, device):
        if self._closed:
            raise RuntimeError('GPU session is closed.')
        if mesh is not self.mesh:
            raise ValueError('GPU session belongs to a different SampleMesh.')
        if cuda_device(device) != self.device:
            raise ValueError('GPU session and requested device differ.')

    def _wait_ready(self):
        torch.cuda.current_stream(self.device).wait_event(self._ready)

    def prepare_rays(self, origin, targets):
        delta = targets - origin
        initial = delta / (delta.norm(dim=1, keepdim=True) + 1e-8)
        double = initial.double()
        norm = torch.sqrt(double[:, 0]*double[:, 0] + double[:, 1]*double[:, 1] + double[:, 2]*double[:, 2])
        # Keep the validated second normalization. Give coincident targets a safe
        # direction; the distance rule will classify them as visible.
        direction = (double / norm.clamp_min(1e-30)[:, None]).float()
        direction[:, 0] = torch.where(norm > 0, direction[:, 0], 1.)
        native_origin = ((origin.double() - self.offset_gpu) * self.scale).float()
        return native_origin.contiguous(), direction.contiguous()

    def visibility(self, origin, targets, source_faces):
        native_origin, directions = self.prepare_rays(origin, targets)
        ids, distance = self.scene.trace(native_origin, directions)
        delta = targets - origin
        length = torch.sqrt(delta[:, 0]*delta[:, 0] + delta[:, 1]*delta[:, 1] + delta[:, 2]*delta[:, 2])
        return (ids < 0) | ((ids >= 0) & (ids == source_faces)) | (distance.double()/self.scale > (length - 1e-4).double())

    def fov(self, camera, pyramid):
        from emtdicompute.camera_handler.camera import pyramid_masking
        self.check(self.mesh, self.device)
        with torch.cuda.device(self.device):
            self._wait_ready()
            return pyramid_masking(self.sampled_mesh, camera, pyramid, device=self.device)

    def prelims(self, camera, mask):
        self.check(self.mesh, self.device)
        if mask.shape != (len(camera.cam_orig), len(self.points)):
            raise ValueError('FOV mask must have shape (number of viewpoints, number of sampled points).')
        with torch.cuda.device(self.device):
            self._wait_ready()
            origins = torch.as_tensor(camera.cam_orig, dtype=torch.float32, device=self.device)
            mask = mask.to(device=self.device, dtype=torch.bool)
            ps, ls, ns, fs = [], [], [], []
            for frame in range(len(origins)):
                candidates = torch.nonzero(mask[frame], as_tuple=True)[0]
                visible = self.visibility(origins[frame], self.points[candidates], self.faces[candidates])
                selected = candidates[visible]
                ps.append(self.points[selected])
                ls.append(self.labels[selected])
                ns.append(self.normals[selected])
                fs.append(torch.full((len(selected),), frame, dtype=torch.long, device=self.device))
            if not len(origins):
                return Em3diPrelims(self.points[:0], self.labels[:0], self.normals[:0],
                                   torch.empty(0, dtype=torch.long, device=self.device), 0)
            return Em3diPrelims(torch.cat(ps), torch.cat(ls), torch.cat(ns), torch.cat(fs), len(origins))

    def close(self):
        if not self._closed:
            if self.scene is not None:
                self.scene.close()
            self._closed = True

    def __enter__(self):
        self.check(self.mesh, self.device)
        return self

    def __exit__(self, *args):
        self.close()
