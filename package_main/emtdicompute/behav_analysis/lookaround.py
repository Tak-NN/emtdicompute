import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
import shutil
import inspect
import open3d as o3d

from emtdicompute.utils import debug_templates
from emtdicompute.model_analysis import field
from emtdicompute.em3di_visualization import visualizer


# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



_OUTPUT_LA_FULL_DEFAULT = './outputs/lookaround/full'
_OUTPUT_LA_ADJ_DEFAULT = './outputs/lookaround/adjacent'
_OUTPUT_GRID_OUTPUT_DEFAULT = './outputs/la_em3di_unified'

def compute_la_value(
        records_dir: str,
        remove_exist_output = False,
        output_full_dir: str = None,
        output_adj_dir: str = None,
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    """
    outputs:
        - 
    """

    extension = "/*csv"
    records = records_dir + extension

    if output_full_dir is None:
        output_full_dir = Path(_OUTPUT_LA_FULL_DEFAULT)
    else:
        output_full_dir = Path(output_full_dir)

    if output_adj_dir is None:
        output_adj_dir = Path(_OUTPUT_LA_ADJ_DEFAULT)
    else:
        output_adj_dir = Path(output_adj_dir)
    

    if Path(output_full_dir).exists() and remove_exist_output:
        shutil.rmtree(output_full_dir, ignore_errors = True)
    if Path(output_adj_dir).exists() and remove_exist_output:
        shutil.rmtree(output_adj_dir, ignore_errors = True)

    os.makedirs(output_full_dir, exist_ok=True)

    if not Path(records_dir).exists():
        raise FileNotFoundError(f"Record directory not found: {records_dir}")
    
    record_files = glob.glob(records)
    
    cols_pos = ['Position_X', 'Position_Y', 'Position_Z']
    cols_dir = ['HeadDir_X', 'HeadDir_Y', 'HeadDir_Z']

    for record in record_files:
        filename = os.path.basename(record)
        print(record)
        df_rec = pd.read_csv(record)

        df_rec_pos = df_rec.loc[:, cols_pos]

        df_traj_adjacents = df_rec_pos[(df_rec_pos[cols_pos] != df_rec_pos[cols_pos].shift(1)).any(axis = 'columns')].copy()

        pos_prev = df_traj_adjacents.loc[:, cols_pos].shift(1).to_numpy()
        pos_current = df_traj_adjacents.loc[:, cols_pos].to_numpy()
        pos_next = df_traj_adjacents.loc[:, cols_pos].shift(-1).to_numpy()

        traj_tangents = pos_next - pos_prev # trajectory tangent on the pos_current location
        traj_tangents[0] = pos_next[0] - pos_current[0] # because the starting position does not have "previous" point.
        traj_tangents[-1] = pos_current[-1] - pos_prev[-1] # because the ending position does not have "next" point.
        traj_tangents[:, 1] = 0.0 # planar (x, 0, z)
        norm_tangents = np.linalg.norm(traj_tangents, axis = 1, keepdims = True)
        traj_tangents = traj_tangents / (norm_tangents + 1e-6)

        df_traj_adjacents.loc[:, ['tan_X', 'tan_Y', 'tan_Z']] = traj_tangents
        df_traj_expanded = df_traj_adjacents.reindex(range(len(df_rec)), method = 'ffill')

        ### --- calculation of angles between directions of camera and tangent
        df_tangents_expanded = df_traj_expanded[['tan_X', 'tan_Y', 'tan_Z']]
        v_tangents_expanded = df_tangents_expanded.to_numpy()
        v_headdir = df_rec.loc[:, cols_dir].to_numpy()
        v_headdir[:, 1] = 0.0 # planar
        norm_v_hd = np.linalg.norm(v_headdir, axis = 1, keepdims=True)
        v_headdir = v_headdir / (norm_v_hd + 1e-6)
        cos = np.sum(v_tangents_expanded * v_headdir, axis = 1)
        ang_tan_hdir = np.arccos(np.clip(cos, -1.0, 1.0))
        ang_tan_hdir = np.rad2deg(ang_tan_hdir)

        ### --- calculation of delta-ang_tan_hdir
        delta_ang = np.abs(np.diff(ang_tan_hdir))
        df_delta_ang = pd.Series(delta_ang)

        ### --- calculation of stopping time and sum_lookaround during stopping
        idx_start_of_stopping = df_traj_adjacents.index.to_numpy()
        stopping_duration = []
        sum_dang_during_stop = []
        delta_ang_per_time = []
        stop_begin_timestamp = []
        
        for i in range(len(idx_start_of_stopping)):
            seg_start = idx_start_of_stopping[i]
            if i < len(idx_start_of_stopping) - 1:
                seg_end = idx_start_of_stopping[i+1]
            else:
                seg_end = len(df_rec) - 1
            
            t_start_stopping = df_rec.loc[seg_start, 'Timestamp']
            t_end_stopping = df_rec.loc[seg_end, 'Timestamp']
            elapsed = t_end_stopping - t_start_stopping

            delta_ang_sum = df_delta_ang[seg_start:seg_end].sum()
            dang_per_time = df_delta_ang[seg_start:seg_end].sum() / (elapsed + 1e-6)

            stopping_duration.append(elapsed)
            sum_dang_during_stop.append(delta_ang_sum)
            delta_ang_per_time.append(dang_per_time)
            stop_begin_timestamp.append(t_start_stopping)


        df_to_save_full = df_rec
        df_to_save_full[['tan_X', 'tan_Y', 'tan_Z']] = df_tangents_expanded
        df_to_save_full['ang_tan_hdir'] = ang_tan_hdir
        df_to_save_full['delta_ang'] = pd.Series(delta_ang).reindex(df_to_save_full.index)

        df_to_save_full = df_to_save_full.round(3)
        df_to_save_full = df_to_save_full.drop(df_to_save_full.index[-1])

        full_filename = "la_full_" + filename
        output_path = os.path.join(output_full_dir, full_filename)
        df_to_save_full.to_csv(output_path, index = True)
        print(f"[file saved] {full_filename}")

        df_to_save_adj = df_traj_adjacents
        df_to_save_adj['stopping_duration'] = stopping_duration
        df_to_save_adj['sum_dang_during_stop'] = delta_ang_per_time
        df_to_save_adj['delta_ang_per_time'] = sum_dang_during_stop
        df_to_save_adj['Timestamp'] = stop_begin_timestamp
        df_to_save_adj = df_to_save_adj.round(3)

        adj_filename = "la_adj_" + filename
        os.makedirs(output_adj_dir, exist_ok=True)
        output_path = os.path.join(output_adj_dir, adj_filename)
        df_to_save_adj.to_csv(output_path, index = True)
        print(f"[file saved] {adj_filename}")



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def grid_analysis(
        la_adj_dir: str | Path,
        em3di_results_dir: str | Path,
        floor_mesh_path: str, 
        output_dir: str | Path = None,
        grid_size: float = 2.0,
        eye_height: float = 1.75,
        eh_tolerance: float = 0.1,
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)
    
    if output_dir is None:
        output_dir = Path(_OUTPUT_GRID_OUTPUT_DEFAULT)
    else:
        output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok = True)
    

    df_grid_viewpts = field.gen_grid_viewpoints(
        floor_mesh_path, grid_size, eye_height, output_grid_csv=True
    ).cam_orig
    print(f"[grid points] shape: {df_grid_viewpts.shape}")

    cen_x = df_grid_viewpts[:, 0] # x
    cen_z = df_grid_viewpts[:, 2] # z
    cen_y = df_grid_viewpts[:, 1] # y
    print(f"[center_x] {cen_x.shape}")

    g_half = grid_size / 2.0
    x_min = cen_x - g_half
    x_max = cen_x + g_half
    z_min = cen_z - g_half
    z_max = cen_z + g_half
    y_min = cen_y - eh_tolerance
    y_max = cen_y + eh_tolerance
    df_grid_frame = pd.DataFrame(
        {'x_min': x_min,
         'x_max': x_max,
         'z_min': z_min,
         'z_max': z_max,
         'y_min': y_min,
         'y_max': y_max,}
    )
    df_grid_frame.to_csv('./outputs/grid_frames.csv')

    def _gen_grid_mask(df_positions):
        x = df_positions['Position_X'].to_numpy()
        z = df_positions['Position_Z'].to_numpy()
        y = df_positions['Position_Y'].to_numpy()

        mask_x = (x[:, None] >= x_min[None, :]) & (x[:, None] < x_max[None, :])
        mask_z = (z[:, None] >= z_min[None, :]) & (z[:, None] < z_max[None, :])
        mask_y = (y[:, None] >= y_min[None, :]) & (y[:, None] < y_max[None, :])
        mask = mask_x & mask_z & mask_y

        print(f"[mask generated] {mask.shape}")

        return mask


    adj_files = sorted(glob.glob(str(Path(la_adj_dir) / "*.csv")))
    la_in_grid = []


    for adj_file in adj_files:
        df_adj = pd.read_csv(adj_file)

        mask_adj = _gen_grid_mask(df_adj)
        df_mask_adj = pd.DataFrame(mask_adj)
        df_mask_adj.to_csv('./outputs/mask.csv', index = False)

        _, G_adj = mask_adj.shape # F for num of frames, G for num of grids

        for g in range(G_adj):
            idx = np.where(mask_adj[:, g])[0] # extract pts inside grid g
            if len(idx) == 0:
                print(f"[grid masking] no points inside grid {g}")
                continue

            stop_sum = df_adj.loc[idx, 'stopping_duration'].sum()
            dang_sum = df_adj.loc[idx, 'sum_dang_during_stop'].sum()
            mean_dang = dang_sum / (stop_sum + 1e-12)

            out = {
                'grid_id': g,
                'center_x': cen_x[g],
                'center_z': cen_z[g],
                'height_y': cen_y[g],
                'stop_dur_sum': stop_sum,
                'la_sum': dang_sum,
                'la_mean_pertime': mean_dang,
            }

            la_in_grid.append(out)


    em3di_met_files = sorted(glob.glob(str(Path(em3di_results_dir) / '*.csv')))
    em3ti_in_grid = []

    cols_em3_metrics = ['vh_ratio', 'td_ratio', 'v_jaggedness']

    for em3di_met_file in em3di_met_files:
        df_em3 = pd.read_csv(em3di_met_file)

        mask_em3 = _gen_grid_mask(df_em3)

        _, G_em3 = mask_em3.shape

        for g in range(G_em3):
            idx = np.where(mask_em3[:, g])[0]
            if len(idx) == 0:
                continue

            mean_em3_metrics = df_em3.loc[idx, cols_em3_metrics].mean().round(3)

            out = {
                'grid_id': g,
                'center_x': cen_x[g],
                'center_z': cen_z[g],
                'height_y': cen_y[g]
            }
            for col in cols_em3_metrics:
                out[f'mean_{col}'] = mean_em3_metrics[col]

            em3ti_in_grid.append(out)
    
    df_output_adj_la = pd.DataFrame(la_in_grid).round(3)
    df_output_mean_em3di = pd.DataFrame(em3ti_in_grid).round(3)
    df_output_merged = pd.merge(df_output_adj_la, df_output_mean_em3di, on = ['grid_id','center_x','center_z','height_y'])
    df_output_merged.to_csv(Path(output_dir) / 'grid_summary_unified.csv', index = False)

    return df_output_merged



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def visualize_grid_unified_bubble(
        df_grid_analysis_merged: pd.DataFrame | str,
        col_to_visualize: str,
        bldg_mesh: str,
        record_csv: str,
        r_scale: float = 1.0,
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    geoms = []

    geom_bldg = visualizer.visualize_bldg(bldg_mesh)
    geom_path = visualizer._path_to_pline(record_csv)

    r_factor = df_grid_analysis_merged[col_to_visualize].to_numpy()
    rf_clipped = np.clip(r_factor, a_min=0, a_max=None)

    rf_clipped_log = np.log1p(rf_clipped)
    r_factor_norm = (rf_clipped_log - rf_clipped_log.min()) / (rf_clipped_log.max() - rf_clipped_log.min() + 1e-12)
    r_factor_positive = r_factor_norm + np.abs(r_factor_norm.min()) + 1e-3

    radii = r_factor_positive * r_scale
    sph_x = df_grid_analysis_merged['center_x']
    sph_z = df_grid_analysis_merged['center_z']
    sph_y = df_grid_analysis_merged['height_y']

    for i, r in enumerate(radii):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius = r)
        sphere.compute_vertex_normals()
        sphere.translate([sph_x[i], sph_y[i], sph_z[i]])
        sphere.paint_uniform_color([0.3, 0.6, 0.6])
        geoms.append(sphere)

    geoms = geoms + geom_bldg + geom_path

    visualizer.open_visualization_window(geoms)

