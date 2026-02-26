import numpy as np
import os
import trimesh
import inspect
import pandas as pd
from pathlib import Path

import emtdicompute.utils.datatypes as dc
from emtdicompute.utils import debug_templates



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



_CACHE_DIR_DEFAULT = Path('./outputs/mesh_sampled')

def _cache_paths(
        mesh_path: Path, 
        sampling_density: int, 
        cache_dir: Path | None = None,
    ):

    if cache_dir is None:
        cache_dir = _CACHE_DIR_DEFAULT
    cache_dir.mkdir(parents = True, exist_ok = True)

    mesh_filename = Path(mesh_path).stem
    cache_filepath = cache_dir / f"ptc_{mesh_filename}_d{sampling_density}.csv"
    meta_filepath = cache_dir / f"ptc_{mesh_filename}_d{sampling_density}_meta.csv"

    return cache_filepath, meta_filepath



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def mesh_sampling(
        mesh_path, 
        sampling_density: float = 100.0,
        cache: bool = True,
        force_recompute: bool = False,
        force_usecache: bool | None = None,
        cache_dir: Path | None = None,
        seed: int | None = None,
    ) -> dc.SampleMesh:
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)
    """
    inputs:
        - mesh_path: str
        - sampling_density: float. number of sampling points per area
    outputs:
        - sampled_points: sampled points on mesh surface ([N, 3] where N is total   number of sampled points)
        - face_vh_labels: vertical / horizontal labels for each sampled point ([N, ]). 0 = horizontal, 1 = vertical, 2 = other
        - face_normals: normal vectors (length = 1) of faces at each sampled point ([N, 3])
    """
    if force_usecache is None:
        force_usecache = bool(input(f'force_usecache? Type "False" or "True"'))

    ### --- process related to cache
    if cache_dir is None:
        cache_dir = _CACHE_DIR_DEFAULT
    cache_ptc_file, meta_file = _cache_paths(mesh_path, sampling_density, cache_dir)

    if cache and not force_recompute and cache_ptc_file.exists() and meta_file.exists():
        meta = pd.read_csv(meta_file).iloc[0].to_dict()
        src_ok = (
            meta.get('src_path') == str(mesh_path)
            and abs(meta.get('src_mtime', -1) - Path(mesh_path).stat().st_mtime) < 1e-6
            and int(meta.get('density', -1)) == sampling_density
            # and int(meta.get('seed')) == seed
        )
        if src_ok or force_usecache:
            df_cached = pd.read_csv(cache_ptc_file)
            sampled_points = df_cached[['ptc_X', 'ptc_Y', 'ptc_Z']].to_numpy()
            face_vh_labels = df_cached['vh_labels'].to_numpy().astype(int)
            face_normals = df_cached[['face_n_X', 'face_n_Y', 'face_n_Z']].to_numpy()
            mesh = trimesh.load(mesh_path, force = 'scene').to_geometry()
            print(f"[sampled points on mesh '{mesh_path}' have been loaded from cache.]")
            return dc.SampleMesh(sampled_points, face_vh_labels, face_normals, mesh, sampling_density)


    ### === main processing (skipped when the proper cache exists)
    mesh = trimesh.load(mesh_path, force='scene').to_geometry()

    vertices = mesh.vertices
    faces = mesh.faces

    ### === sampling process main === ###

    if seed is None:
        seed = int.from_bytes(os.urandom(8), 'big')
    rng = np.random.default_rng(seed)

    ### --- vectors of vertices of each triangle
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    ### --- calc num of sampling points
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    n_points_per_face = np.maximum(np.int32((face_areas * sampling_density)), 1)
    total_sampling_n = n_points_per_face.sum().item()
    print(f"[total n of sampling points]\n{total_sampling_n}\n")

    ### --- sampling
    ## -- set repeated face indices
    face_indices = np.repeat(np.arange(len(faces)), n_points_per_face)
    ## -- set repeated vertices
    tri_v0 = v0[face_indices]
    tri_v1 = v1[face_indices]
    tri_v2 = v2[face_indices]
    ## -- random generation of Barycentric coordinates
    k1 = np.sqrt(rng.random(total_sampling_n))
    k2 = rng.random(total_sampling_n)
    t = 1 - k1
    s = k1 * (1 - k2)
    u = k1 * k2
    ## -- sampled points on faces
    sampled_points = tri_v0*t[:, None] + tri_v1*s[:, None] + tri_v2*u[:, None]

    ### --- vertical / horizontal labeling
    face_normals = mesh.face_normals
    face_normals = face_normals.round(6)
    y_abs = np.abs(face_normals[:, 1])
    labels = np.full(len(faces), "2", dtype=int)
    labels[y_abs >= 0.985] = "0" # horizontal
    labels[y_abs <= 0.10] = "1" # vertical

    ## -- labels at each sampled point
    face_vh_labels = labels[face_indices]
    face_normals = face_normals[face_indices]


    ### === caching
    if cache:
        df_to_cache = pd.DataFrame(
            columns = [
                'ptc_X', 'ptc_Y', 'ptc_Z',
                'vh_labels', 'face_normals'
            ]
        )

        df_to_cache[['ptc_X', 'ptc_Y', 'ptc_Z']] = sampled_points
        df_to_cache['vh_labels'] = face_vh_labels
        df_to_cache[['face_n_X', 'face_n_Y', 'face_n_Z']] = face_normals
        df_to_cache.to_csv(cache_ptc_file, index = False)

        meta = {
            'src_path': str(mesh_path),
            'src_mtime': Path(mesh_path).stat().st_mtime,
            'density': sampling_density,
            'seed': int(seed),
            'ptc_count': total_sampling_n,
        }
        pd.DataFrame([meta]).to_csv(meta_file, index = False)
        print(f"[cacje files saved]\n- {cache_ptc_file}\n- {meta_file}\n")


    return dc.SampleMesh(sampled_points, face_vh_labels, face_normals, mesh, sampling_density)


