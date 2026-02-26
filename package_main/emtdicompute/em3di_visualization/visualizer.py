import pandas as pd
import torch
import trimesh
import inspect
import numpy as np
from emtdicompute.utils import debug_templates
import emtdicompute.utils.datatypes as dc
import open3d as o3d
from emtdicompute.camera_handler import camera
from emtdicompute.scene import sample_mesh
from emtdicompute.embodied_3d_isovist import isovist



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def open_visualization_window(
        geoms: o3d.geometry,
        width = 960,
        height = 720,
        background_color = None,
        camera_lookat = None,
        camera_front = None,
        camera_up = None,
        camera_zoom = None,
):

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = width, height = height)
    render_option = vis.get_render_option()
    if background_color is None:
        background_color = [0.1, 0.1, 0.2]
    render_option.background_color = np.asarray(background_color)

    if isinstance(geoms, (list, tuple)):
        for g in geoms:
            vis.add_geometry(g)
    else:
        vis.add_geometry(geoms)

    if any(v is not None for v in [camera_lookat, camera_front, camera_up, camera_zoom]):
        view_ctl = vis.get_view_control()
        if camera_lookat is not None:
            view_ctl.set_lookat(camera_lookat)
        if camera_front is not None:
            view_ctl.set_front(camera_front)
        if camera_up is not None:
            view_ctl.set_up(camera_up)
        if camera_zoom is not None:
            view_ctl.set_zoom(camera_zoom)
        vis.update_renderer()

    while vis.poll_events():
        vis.update_renderer()

    vis.destroy_window()
    # o3d.visualization.draw_geometries(geoms, width, height)


# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |


def save_visualization_image(
        geoms: o3d.geometry,
        output_path: str,
        width = 960,
        height = 720,
        background_color = None,
        camera_lookat = None,
        camera_front = None,
        camera_up = None,
        camera_zoom = None,
):

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = width, height = height)
    render_option = vis.get_render_option()
    if background_color is None:
        background_color = [0.1, 0.1, 0.2]
    render_option.background_color = np.asarray(background_color)

    if isinstance(geoms, (list, tuple)):
        for g in geoms:
            vis.add_geometry(g)
    else:
        vis.add_geometry(geoms)

    if any(v is not None for v in [camera_lookat, camera_front, camera_up, camera_zoom]):
        view_ctl = vis.get_view_control()
        if camera_lookat is not None:
            view_ctl.set_lookat(camera_lookat)
        if camera_front is not None:
            view_ctl.set_front(camera_front)
        if camera_up is not None:
            view_ctl.set_up(camera_up)
        if camera_zoom is not None:
            view_ctl.set_zoom(camera_zoom)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path, do_render = True)
    vis.destroy_window()


# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def visualize_bldg(
        bldg_mesh_path: str,
        color = None,
):
    geoms_bldg = _mesh_to_plines(bldg_mesh_path, color = color)
    # geoms_bldg = _mesh_to_plines_legacy_allwires(bldg_mesh_path)
    return geoms_bldg



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def visualize_visible_vhfaces_atframe(
        record_path: str,
        bldg_mesh_path: str, 
        density: int = 100, 
        frame: int = 100,
        mode: str = None, # select from 'vh' and 'td'
    ):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)
    
    geoms_vhfaces_atframe = []

    geoms_path_pline = _path_to_pline(record_path)
    em3di_at_frame = _em3di_atframe(record_path, frame, bldg_mesh_path, density = density)
    geoms_points_vhpoints_atframe = _em3di_atframe_visualizer(em3di_at_frame, mode=mode)
    geoms_camera_origin = _camera_atframe_visualizer(record_path, frame)

    geoms_vhfaces_atframe = geoms_path_pline + geoms_points_vhpoints_atframe + geoms_camera_origin

    return geoms_vhfaces_atframe



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def visualize_em3diprops_along_trajectory(
        df_em3di_props: pd.DataFrame | str,
        mode: str = None,
        log_normalized_color = True,
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    geoms_em3diprops_alongtraj = []

    geoms_trajectory_pline = _path_to_pline(df_em3di_props, color = [0.7, 0.7, 0.7])
    geoms_traj_gradient_pts = _gradient_camorig_ptcloud(df_em3di_props, mode = mode, log_normalized_color = log_normalized_color)

    geoms_em3diprops_alongtraj = geoms_trajectory_pline + geoms_traj_gradient_pts

    return geoms_em3diprops_alongtraj



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def _gradient_camorig_ptcloud(
        df_em3di_props: pd.DataFrame | str, 
        mode: str = None,
        log_normalized_color = True
    ):

    """
    inputs:
        - df_em3di_props: 
    outputs:
        - geoms_ptcloud: open3d pointcloud.
            - by default (mode is None), this pointcloud is colored gray.
            - by setting mode (vh or td), this pointcloud will be colored gradiently according to the em3di_props (vh_ratio or td_ratio)
    """
    geoms_pts = []

    # read dataframe from csv path or DataFrame output of field_analysis_em3diprops()
    if isinstance(df_em3di_props, str):
        df_em3di_props = pd.read_csv(df_em3di_props)
    elif isinstance(df_em3di_props, pd.DataFrame):
        df_em3di_props = df_em3di_props

    viewpoints = df_em3di_props[['Position_X', 'Position_Y', 'Position_Z']].to_numpy()
    viewpts_base = o3d.geometry.PointCloud()
    viewpts_base.points = o3d.utility.Vector3dVector(viewpoints)

    # set color gradient
    n_of_viewpts = len(viewpoints)
    if mode is None:
        print('[visualization] no mode specified. plain color will be attached.')
        colors = np.full((n_of_viewpts, 3), [0.0, 1.0, 0.0])
    elif mode == 'vh':
        gradient_factor = df_em3di_props['vh_ratio'].to_numpy().astype(float)
    elif mode == 'td':
        gradient_factor = df_em3di_props['td_ratio'].to_numpy().astype(float)
    elif mode == 'vj':
        gradient_factor = df_em3di_props['v_jaggedness'].to_numpy().astype(float)
    elif mode == 'vv':
        gradient_factor = df_em3di_props['visible_volume'].to_numpy().astype(float)

    g_factor = gradient_factor
    g_factor_clipped = np.clip(g_factor, a_min = 0, a_max = None)

    if not log_normalized_color:
        # linear normalization (0 to 1)
        gradient_factor_norm = (g_factor_clipped - g_factor_clipped.min()) / (g_factor_clipped.max() - g_factor_clipped.min() + 1e-12)

    elif log_normalized_color:
        # log normalization
        gradient_factor_log = np.log1p(g_factor_clipped)
        gradient_factor_norm = (gradient_factor_log - gradient_factor_log.min()) / (gradient_factor_log.max() - gradient_factor_log.min() + 1e-12)
    
    colors = np.vstack([
        gradient_factor_norm,
        np.zeros_like(gradient_factor_norm) + 1.0,
        # np.zeros_like(gradient_factor_norm)
        1 - gradient_factor_norm
    ]).T

    viewpts_base.colors = o3d.utility.Vector3dVector(colors)
    geoms_pts.append(viewpts_base)

    return geoms_pts
    


# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def _path_to_pline(record: str | pd.DataFrame, color: np.ndarray = None):

    geoms_path_pline = []

    if isinstance(record, str):
        df_rec = pd.read_csv(record)
    elif isinstance(record, pd.DataFrame):
        df_rec = record
    c_orig, _ = camera.extract_camera_from_record(df_rec)

    # ensure open trajectory: drop duplicated last=first point if present
    traj_points = c_orig
    if (
        isinstance(c_orig, np.ndarray)
        and c_orig.ndim == 2
        and c_orig.shape[0] >= 2
        and np.allclose(c_orig[0], c_orig[-1])
    ):
        traj_points = c_orig[:-1]

    # draw trajectory by connecting points from record file
    if isinstance(traj_points, np.ndarray) and traj_points.ndim == 2 and traj_points.shape[0] >= 2:
        num_pts = traj_points.shape[0]
        lines = np.column_stack((np.arange(num_pts - 1, dtype=np.int32),
                                 np.arange(1, num_pts, dtype=np.int32)))
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(traj_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        if color is None:
            color = [1.0, 0.2, 1.0] # purple by default
        elif color is not None:
            color = color
        line_set.colors = o3d.utility.Vector3dVector(
            np.repeat([color], repeats=lines.shape[0], axis=0)
        )
        geoms_path_pline.append(line_set)

    return geoms_path_pline



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def _mesh_to_plines_legacy_allwires(obj_path: str, color: np.ndarray = None):
    geoms_obj_plines = []

    scene = trimesh.load(obj_path, force='scene')
    mesh = scene.to_geometry()

    trimesh_list = []
    if mesh is None:
        trimesh_list = []
    elif isinstance(mesh, (list, tuple)):
        trimesh_list = list(mesh)
    elif isinstance(mesh, dict):
        trimesh_list = list(mesh.values())
    else:
        trimesh_list = [mesh]

    for tm in trimesh_list:
        if not isinstance(tm, trimesh.Trimesh):
            continue
        if tm.vertices is None or tm.faces is None or len(tm.vertices) == 0 or len(tm.faces) == 0:
            continue

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
        o3d_mesh.compute_vertex_normals()

        # Draw OBJ as wireframe (visually transparent) while keeping others solid
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
        if len(line_set.lines) > 0:
            if color is None:
                color = [0.8, 0.8, 0.8] # light gray by default
            if color is not None:
                color = color
            line_colors = np.repeat([color], repeats=len(line_set.lines), axis=0)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
        geoms_obj_plines.append(line_set)

    return geoms_obj_plines



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def _mesh_to_plines(obj_path: str, color: np.ndarray = None, angle_deg: float = 1.0):
    geoms_obj_plines = []
    scene = trimesh.load(obj_path, force='scene')
    meshes = scene.to_geometry()
    
    if isinstance(meshes, (list, tuple, dict)):
        trimesh_list = meshes
    else:
        trimesh_list = [meshes]
    # trimesh_list = meshes if isinstance(meshes, (list, tuple, dict)) else [meshes]

    for tm in trimesh_list:
        if not isinstance(tm, trimesh.Trimesh) or tm.vertices is None or tm.faces is None:
            continue

        crease_th = np.deg2rad(angle_deg)
        print(tm.vertices)


        # --- boundary edges: count how many faces share each edge ---
        face_edges = np.sort(np.vstack([
            tm.faces[:, [0, 1]],
            tm.faces[:, [1, 2]],
            tm.faces[:, [2, 0]],
        ]), axis=1)
        print('------')
        print(f"[face edges] {face_edges}")
        unique_edges, counts = np.unique(face_edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        print('------')
        print(f"[boundary edges] {unique_edges}")

        # --- sharp edges: adjacent faces whose normals differ over threshold ---
        adj_edges = tm.face_adjacency_edges          # (n_adj, 2)
        adj_angles = tm.face_adjacency_angles        # (n_adj,)
        sharp_edges = adj_edges[adj_angles > crease_th]

        all_edges = np.vstack([boundary_edges, sharp_edges])
        all_edges = np.unique(np.sort(all_edges, axis=1), axis=0)
        if len(all_edges) == 0:
            continue

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(tm.vertices)
        line_set.lines = o3d.utility.Vector2iVector(all_edges.astype(np.int32))
        if color is None:
            line_color = [0.8, 0.8, 0.8]
        if color is not None:
            line_color = color
        line_set.colors = o3d.utility.Vector3dVector(
            np.repeat([line_color], repeats=len(all_edges), axis=0)
        )
        geoms_obj_plines.append(line_set)

    return geoms_obj_plines




# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def _camera_atframe(record_path: str, frame: int):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    df_rec = pd.read_csv(record_path)
    c_orig, c_forward = camera.extract_camera_from_record(df_rec)

    c_orig_atframe = c_orig[frame:frame+2]
    c_forward_atframe = c_forward[frame:frame+2]
    print(f"frame: {frame}, {c_orig_atframe}")
    camera_atframe = camera.combine_camera_pos_and_dir(c_orig_atframe, c_forward_atframe)
    print(camera_atframe)

    return camera_atframe



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def _em3di_atframe(record_path: str, frame: int, mesh: dc.SampleMesh, density = 100):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    sampled_mesh = sample_mesh.mesh_sampling(mesh, sampling_density = density)

    camera_atframe = _camera_atframe(record_path, frame)
    pyramid_atframe = camera.camera_pyramids(camera_atframe)
    pyramid_mask_atframe = camera.pyramid_masking(sampled_mesh, camera_atframe, pyramid_atframe, device = 'cpu')
    em3di_output = isovist.embodied_3d_isovist_prelims(sampled_mesh, camera_atframe, pyramid_mask_atframe, device = 'cpu')

    print(pyramid_atframe.pyram_points_allframe)

    return em3di_output



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def _em3di_atframe_visualizer(
        em3di_output: dc.Em3diPrelims,
        mode: str = None, # select from 'vh' and 'td'
    ):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    geoms_labeled_points = []

    visible_pts = em3di_output.visible_points_allframes.detach().cpu().numpy()
    normals = em3di_output.normals_visipts_allframes.detach().cpu().numpy()
    vhlbls = em3di_output.vhlbls_visipts_allframes.detach().cpu().numpy()
    frame_indices = em3di_output.frame_indices.detach().cpu().numpy()
    
    ## extract the specified frame
    frame_indices = ~frame_indices.astype(bool)
    visible_pts_atframe = visible_pts[frame_indices]
    normals_atframe = normals[frame_indices]
    vhlbls_atframe = vhlbls[frame_indices]
    print(f"vh labels: {set(vhlbls_atframe)}")

    if mode != 'vh' and mode != 'td':
        print(f'[mode error] please select mode from "vh" and "td"')

    elif mode is None or mode == "vh": 
        ## point cloud for H face points
        h_mask = ~vhlbls_atframe.astype(bool)
        pts_on_hfaces = visible_pts_atframe[h_mask]
        h_pts_base = o3d.geometry.PointCloud()
        h_pts_base.points = o3d.utility.Vector3dVector(pts_on_hfaces)
        h_pts_base.paint_uniform_color([1.0, 0.3, 0.4])
        geoms_labeled_points.append(h_pts_base)

        ## point cloud for V face points
        v_mask = vhlbls_atframe.astype(bool)
        pts_on_vfaces = visible_pts_atframe[v_mask]
        v_pts_base = o3d.geometry.PointCloud()
        v_pts_base.points = o3d.utility.Vector3dVector(pts_on_vfaces)
        v_pts_base.paint_uniform_color([0.3, 1.0, 0.4])
        geoms_labeled_points.append(v_pts_base)

    elif mode == "td":
        h_mask = ~vhlbls_atframe.astype(bool)
        t_mask = h_mask & (normals_atframe[:, 1] < 0)
        d_mask = h_mask & (normals_atframe[:, 1] > 0)

        t_pts = visible_pts_atframe[t_mask]
        t_pts_base = o3d.geometry.PointCloud()
        t_pts_base.points = o3d.utility.Vector3dVector(t_pts)
        t_pts_base.paint_uniform_color([1.0, 0.3, 0.4])
        geoms_labeled_points.append(t_pts_base)

        d_pts = visible_pts_atframe[d_mask]
        d_pts_base = o3d.geometry.PointCloud()
        d_pts_base.points = o3d.utility.Vector3dVector(d_pts)
        d_pts_base.paint_uniform_color([0.3, 1.0, 0.4])
        geoms_labeled_points.append(d_pts_base)


    return geoms_labeled_points



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def _camera_atframe_visualizer(record_path: str, frame: int):
    
    geoms = []

    camera_atframe = _camera_atframe(record_path, frame)
    c_orig_atframe = camera_atframe.cam_orig

    ### draw origin of camera
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius = .05)
    sphere.compute_vertex_normals()
    position = c_orig_atframe[0]
    print("POSITION:",position)
    sphere.translate(position) 
    sphere.paint_uniform_color([0.4,0.6,0.6])

    geoms.append(sphere)

    ### draw pyramid
    pyramid_points = camera.camera_pyramids(camera_atframe, pyram_height=1).pyram_points_allframe[0]
    print(pyramid_points)
    pyram_lines = [
        [4, 0], [4, 1], [4, 2], [4, 3],
        [0, 1], [1, 2], [2, 3], [3, 0]
    ]
    lines_base = o3d.geometry.LineSet()
    lines_base.points = o3d.utility.Vector3dVector(pyramid_points)
    lines_base.lines = o3d.utility.Vector2iVector(pyram_lines)
    lines_base.paint_uniform_color([.94,.46,.18])
    geoms.append(lines_base)

    return geoms
