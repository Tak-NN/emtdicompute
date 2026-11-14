import torch
import numpy as np
import os
import trimesh
import open3d as o3d
import inspect
import pandas as pd
from pathlib import Path

from emtdicompute.em3di_visualization import visualizer
from emtdicompute.utils import datatypes as dc
from emtdicompute.batch import pipeline
from emtdicompute.utils import debug_templates



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |


def split_mesh_by_floor_height(mesh: trimesh.Trimesh, height_tol=1e-3):
    verts = mesh.vertices
    faces = mesh.faces

    ys = verts[:, 1]
    
    rounded = np.round(ys / height_tol) * height_tol
    uniq_heights = np.unique(rounded)

    floor_meshes = []

    for h in uniq_heights:
        # --- mask unique height
        mask_v = np.abs(ys - h) < height_tol
        if not np.any(mask_v):
            continue

        idx_map = {old: new for new, old in enumerate(np.where(mask_v)[0])}

        ### --- extract points with the unique height
        mask_f = np.all(mask_v[faces], axis=1)
        faces_sub = faces[mask_f]
        if len(faces_sub) == 0:
            continue

        faces_new = np.vectorize(idx_map.get)(faces_sub)
        verts_new = verts[mask_v]

        floor_mesh = trimesh.Trimesh(vertices=verts_new, faces=faces_new, process=False)
        floor_meshes.append(floor_mesh)

    return floor_meshes



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |


def gen_grid_viewpoints(
        floor_mesh_path: str,
        grid_size: float = 2,
        eye_height: float = 1.6,
        output_grid_csv = True,
        output_dir: Path = './outputs/walkable_grid',
) -> dc.CameraPosDir:
    """
    outputs:
        - pts_on_floor: grid viewpoints inside the floor boundary
    """

    floor_scene = trimesh.load(floor_mesh_path, force = 'scene')
    print('geometries:', floor_scene.geometry.keys())
    print('count:', len(floor_scene.geometry))

    if len(floor_scene.geometry) == 1:
        mesh = next(iter(floor_scene.geometry.values()))
        floor_meshes = split_mesh_by_floor_height(mesh, height_tol = 1e-3)
    else:
        floor_meshes = list(floor_scene.geometry.values())

    print(f"[floor count detected] {len(floor_meshes)}")


    all_viewpoints = []
    all_dirs = []

    ## generate grid points
    for i, mesh in enumerate(floor_meshes):
        print(f"Floor {i} processing...")
        boundary_corners = mesh.bounds

        min_corner = boundary_corners[0]
        max_corner = boundary_corners[1]

        floor_height = boundary_corners[0][1]

        xs = np.arange(min_corner[0] + (grid_size / 2.0), max_corner[0], grid_size)
        zs = np.arange(min_corner[2] + (grid_size / 2.0), max_corner[2], grid_size)

        grid_x, grid_z = np.meshgrid(xs, zs)
        pts_all_grid = np.vstack([
            grid_x.ravel(),
            np.full(grid_x.size, floor_height),
            grid_z.ravel(),
        ]).T

        ## cull points outside floor
        prox_query = trimesh.proximity.ProximityQuery(mesh)
        _, dists, _ = prox_query.on_surface(pts_all_grid)

        mask_on_floor = dists < 1e-4
        viewpoints_on_floor = pts_all_grid[mask_on_floor]

        viewpoints_above_floor = viewpoints_on_floor + [0.0, eye_height, 0.0]

        # print(f"[generated viewpoints] \n {viewpoints_above_floor}")
        print(f"[grid viewpoints generated] shape: {viewpoints_above_floor.shape}")

        ## dummy camera direction [0, -1, 0]
        v_down = np.array([1.0, 1.0, 0.0])
        dummy_camera_dirs = np.tile(v_down, (len(viewpoints_above_floor), 1))

        all_viewpoints.append(viewpoints_above_floor)
        all_dirs.append(dummy_camera_dirs)

    if len(all_viewpoints) == 0:
        raise ValueError("no valid floor points detected in any mesh.")
    
    viewpoints = np.vstack(all_viewpoints).round(3)
    dummy_camdirs = np.vstack(all_dirs).round(3)

    # df_viewoins_above_floor = pd.DataFrame(viewpoints_above_floor, columns = ['Position_X', 'Position_Y', 'Position_Z'])

    # empty_timestamp = np.full(1, np.nan)

    if output_grid_csv:
        df = pd.DataFrame(
            viewpoints,
            columns=['Position_X', 'Position_Y', 'Position_Z']
        )
        meshname = Path(floor_mesh_path).stem
        output_filename = f"grid_{meshname}.csv"
        output_path = Path(output_dir) / output_filename
        Path(output_dir).mkdir(parents=True, exist_ok = True)
        df.to_csv(output_path, index = False)
        print(f"[file saved] {output_path}")
        # df_viewoins_above_floor.to_csv('./outputs/field_analysis/viewpoints.csv', index = False)

    return dc.CameraPosDir(viewpoints, dummy_camdirs)



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def gen_dummy_pyarmid_mask(
        mesh_sampling_outputs: dc.SampleMesh,
        dummy_camera: dc.CameraPosDir,
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    """
    outputs:
        - dummy pyramid mask: 
        - Full of True mask ([F, N] where F represents the number of viewpoints_above_floor.
            In the processing, F works as a "number of frames (= trajectory length or num of rows of a record file)".)
        - "Full of True" means that the cameras at each viewpoint have 360 degree field of view. 
    """
    viewpts_abv_floor = dummy_camera.cam_orig
    n_of_viewpts = len(viewpts_abv_floor)
    n_of_sampledpts = len(mesh_sampling_outputs.sampled_points)
    print(f"number of grid vewipoints is {n_of_viewpts}")

    dmy_pyramid_mask = torch.ones((n_of_viewpts, n_of_sampledpts)).bool()
    print("DUMMY PYRAMID GENERATED")
    print(dmy_pyramid_mask)

    return dmy_pyramid_mask



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def gen_dummy_timestamps(
        dummy_camera: dc.CameraPosDir,
):
    """
    outputs:
        - dummy timestamp: 0 to (F-1) corresponding to each viewpoint
    """
    dummy_timestamp = np.arange(0, len(dummy_camera.cam_orig), 1)
    print("dummy timestamps (idx of grid pts) have been generated.")
    return dummy_timestamp



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



_FIELD_ANALYSIS_OUTPUT_DEFAULT = './outputs/em3dimetrics/env_grid_analysis'
def field_analysis_em3dimets(
        mesh_sampling_outputs: dc.SampleMesh,
        dummy_camera: dc.CameraPosDir,
        device = 'cuda',
        output_dir: str = None,
        output_filename: str = None,
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    if output_dir is None:
        output_dir = Path(_FIELD_ANALYSIS_OUTPUT_DEFAULT)
    else:
        output_dir = Path(output_dir)

    if not Path(output_dir).exists():
        os.makedirs(output_dir, exist_ok=True)

    viewpoints_indices = gen_dummy_timestamps(dummy_camera)

    df_field_em3dimets = pipeline.compute_em3di_metrics_batch(mesh_sampling_outputs, 
                                                             dummy_camera, 
                                                             chunk_size = 10, 
                                                             device = device, 
                                                             timestamps = viewpoints_indices,
                                                             save_each_chunk = False,
                                                             field_analysis = True)
    
    if output_filename is None:
        output_filename = output_dir / f"field_results.csv"
    else:
        output_filename = output_dir / output_filename
    df_field_em3dimets.to_csv(output_filename, index = False)
    
    return df_field_em3dimets



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def visualize_viewpoints(
        dummy_camera: dc.CameraPosDir,
        bldg_mesh_path: str = None,
        floor_mesh_path: str = None,
        visualization = False
):

    geoms = []

    # floor mesh
    if floor_mesh_path is not None:
        geoms += visualizer._mesh_to_plines(floor_mesh_path)

    ### --- group and color by floor height
    pts = dummy_camera.cam_orig
    ys = pts[:, 1]

    rounded = np.round(ys / 1e-3) * 1e-3
    uniq_heights = np.unique(rounded)

    base_colors = [
        [0.2, 1.0, 0.2],   # floor 1
        [0.1, 0.8, 0.1],   # floor 2
        [0.05, 0.6, 0.05], # floor 3
        [0.0, 0.4, 0.0],   # floor 4
    ]

    ### --- ptcloud for each floor
    geoms_pts = []
    for i, h in enumerate(uniq_heights):
        mask = (rounded == h)
        pts_floor = pts[mask]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts_floor)

        color = base_colors[i % len(base_colors)]
        pc.paint_uniform_color(color)

        geoms_pts.append(pc)

        print(f"[visualize] floor {i}, height={h:.4f}, count={len(pts_floor)}")

    geoms += geoms_pts

    # bldg mesh
    if bldg_mesh_path is not None:
        geoms += visualizer._mesh_to_plines(bldg_mesh_path)

    # display
    if visualization:
        o3d.visualization.draw_geometries(
            geoms,
            width=960,
            height=720
        )

    return geoms



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def visualize_field_analysis(
        field_analysis_em3dimets: str | pd.DataFrame, 
        trajectory_analysis_em3dimets: str | pd.DataFrame = None,
        bldg_mesh_path: str = None,
        floor_mesh_path: str = None,
        mode: str = None,
        log_normalized_color = True,
        camera_lookat = None,
        camera_front = None,
        camera_up = None,
        camera_zoom = None,
        save_img: bool | None = False,
        img_output_path: str = None,
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)
    """
     - input:
        - field_analysis_em3diprops: string for csv file path. DataFrame for output of field_analysis_em3diprops()
        - mode: select from "vh", "td", "vj"
        - camera_*: optional open3d view control parameters
    """
    geoms = []

    geoms_bldg = []
    if bldg_mesh_path is not None:
        geoms_bldg = visualizer._mesh_to_plines(bldg_mesh_path, color = [0.8, 0.8, 0.8])
    
    geoms_floor = []
    if floor_mesh_path is not None:
        geoms_floor = visualizer._mesh_to_plines(floor_mesh_path)

    ### coloring viewoitns by em3di props
    geoms_pts = visualizer._gradient_camorig_ptcloud(field_analysis_em3dimets, mode = mode)

    if trajectory_analysis_em3dimets is not None:
        geoms_traj_pts = visualizer._gradient_camorig_ptcloud(trajectory_analysis_em3dimets, mode = mode)
    else:
        geoms_traj_pts = []
    geoms = geoms_bldg + geoms_floor + geoms_pts + geoms_traj_pts

    visualizer.open_visualization_window(
        geoms,
        camera_lookat = camera_lookat,
        camera_front = camera_front,
        camera_up = camera_up,
        camera_zoom = camera_zoom,
    )

    if save_img:
        if img_output_path is None:
            img_output_path = '.field_analysis.png'
        
        visualizer.save_visualization_image(
            geoms,
            img_output_path,
            camera_lookat = camera_lookat,
            camera_front = camera_front,
            camera_up = camera_up,
            camera_zoom = camera_zoom,
        )

    return geoms
