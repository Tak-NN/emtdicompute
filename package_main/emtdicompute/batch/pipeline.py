import numpy as np
import open3d as o3d
import torch
import pandas as pd
import os
from pathlib import Path
import time
import shutil
import inspect

from emtdicompute.scene import sample_mesh
from emtdicompute.camera_handler import camera
from emtdicompute.embodied_3d_isovist import isovist
import emtdicompute.embodied_3d_isovist as em3di
import emtdicompute.utils.datatypes as dc
from emtdicompute.utils import debug_templates
from emtdicompute.model_analysis import field




_TEMP_DEFAULT = './outputs/em3di_metrics/temp'



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def _save_output(
    df_em3di_props: pd.DataFrame,
    output_path: str | Path,
):
    
    df_em3di_props.to_csv(output_path, index = False)
    print(f"[File saved]: {output_path}")



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def compute_em3di_metrics(
    mesh: dc.SampleMesh,
    combined_camera: dc.CameraPosDir,
    hfov = 90,
    vfov = 60,
    device = 'cuda',
    timestamps = None,
    field_analysis = False,
    metrics: list[str] | tuple[str, ...] | None = None,
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    """
    output:
        - df_to_save: pd.DataFrame whose metric columns are a subset of [
            'vh_ratio', 
            'td_ratio',
            'v_jaggedness',
            'visible_volume',
            'Position_X', 'Position_Y', 'Position_Z', 
            'HeadDir_X', 'HeadDir_Y', 'HeadDir_Z',
            'Timestamp'(if Timestamps is not None)
        ]
    params:
        - metrics: Optional list/tuple to select metrics to compute from
          ['vh_ratio', 'td_ratio', 'v_jaggedness', 'visible_volume'].
          Defaults to all.
    """

    available_metrics = ("vh_ratio", "td_ratio", "v_jaggedness", "visible_volume")
    if metrics is None:
        metrics_to_compute = list(available_metrics)
    else:
        metrics_to_compute = list(dict.fromkeys(metrics))
        unknown = set(metrics_to_compute) - set(available_metrics)
        if unknown:
            raise ValueError(f"Unknown metrics requested: {sorted(unknown)}")
        if not metrics_to_compute:
            raise ValueError("At least one metric must be requested.")

    t0_process = time.perf_counter()

    print(f"camera origins:\n {combined_camera.cam_orig}")
    print(f"camera forwards:\n {combined_camera.cam_forward}")

    pyramid = camera.camera_pyramids(combined_camera, hfov = hfov, vfov = vfov)

    if field_analysis:
        pyramid_mask = field.gen_dummy_pyarmid_mask(mesh, combined_camera) # combined_camera is expected to be dummy camera when field analysis
    else:
        pyramid_mask = camera.pyramid_masking(mesh, combined_camera, pyramid, device = device)

    em3di_prelims = isovist.embodied_3d_isovist_prelims(mesh, combined_camera, pyramid_mask, device = device)

    df_em3di_props_output = {}
    if "vh_ratio" in metrics_to_compute:
        vh_ratio = isovist.em3disovist_vh_ratio(em3di_prelims, combined_camera, device = device)
        df_em3di_props_output["vh_ratio"] = vh_ratio.detach().cpu().numpy()
    if "td_ratio" in metrics_to_compute:
        td_ratio = isovist.em3disovist_td_ratio(em3di_prelims, combined_camera, device = device)
        df_em3di_props_output["td_ratio"] = td_ratio.detach().cpu().numpy()
    if "v_jaggedness" in metrics_to_compute:
        v_jaggedness = isovist.em3disovist_v_jaggedness(em3di_prelims, combined_camera, mesh, device = device)
        df_em3di_props_output["v_jaggedness"] = v_jaggedness.detach().cpu().numpy()
    if "visible_volume" in metrics_to_compute:
        visible_volume = isovist.em3disovist_visible_volume(em3di_prelims, combined_camera, mesh, device = device)
        df_em3di_props_output["visible_volume"] = visible_volume.detach().cpu().numpy()

    df_em3di_props_output['Position_X'] = combined_camera.cam_orig[:, 0]
    df_em3di_props_output['Position_Y'] = combined_camera.cam_orig[:, 1]
    df_em3di_props_output['Position_Z'] = combined_camera.cam_orig[:, 2]
    df_em3di_props_output['HeadDir_X'] = combined_camera.cam_forward[:, 0]
    df_em3di_props_output['HeadDir_Y'] = combined_camera.cam_forward[:, 1]
    df_em3di_props_output['HeadDir_Z'] = combined_camera.cam_forward[:, 2]

    if timestamps is not None:
        timestamps_array = np.asarray(timestamps)
        expected_len = len(next(iter(df_em3di_props_output.values())))
        if len(timestamps_array) != expected_len:
            raise ValueError("Length of timestamps does not match metric outputs.")
        df_em3di_props_output["Timestamp"] = timestamps_array

    df_to_save = pd.DataFrame(df_em3di_props_output)

    elapsed_time_for_process = time.perf_counter() - t0_process
    print(f"[em3disovist] Total processing time: {elapsed_time_for_process:.3f} s")

    return df_to_save



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def compute_em3di_metrics_batch(
    mesh: dc.SampleMesh,
    combined_camera: dc.CameraPosDir,
    chunk_size: int,
    hfov = 90,
    vfov = 60,
    device = 'cuda',
    timestamps = None,
    temp_dir: str = None,
    save_each_chunk = True,
    field_analysis = False,
    metrics: list[str] | tuple[str, ...] | None = None,
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)
    
    results_chunked = []
    n_of_camrec_rows = combined_camera.cam_orig.shape[0]

    ## delete remaining temp
    if save_each_chunk:
        if temp_dir is None:
            temp_dir = Path(_TEMP_DEFAULT)
        else:
            temp_dir = Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors = True)
        temp_dir.mkdir(parents = True, exist_ok = True)

    for start in range(0, n_of_camrec_rows, chunk_size):
        end = min(start + chunk_size, n_of_camrec_rows)

        chunked_cam_orig = combined_camera.cam_orig[start:end]
        chunked_cam_forward = combined_camera.cam_forward[start:end]
        if len(chunked_cam_orig) == 0:
            continue
        chunked_cam_records = dc.CameraPosDir(chunked_cam_orig, chunked_cam_forward)

        if timestamps is not None:
            timestamps_array = np.asarray(timestamps) 
            chunked_timestamps = timestamps_array[start:end]
        else:
            chunked_timestamps = None

        start_label = start + 1
        end_label = end + 1
        print(f"[batch em3di compute] Processing rows {start_label}-{end_label}")


        chunk_result = compute_em3di_metrics(
            mesh,
            chunked_cam_records,
            hfov = hfov,
            vfov = vfov,
            device = device,
            timestamps = chunked_timestamps,
            field_analysis = field_analysis,
            metrics = metrics,
        )
        results_chunked.append(chunk_result)

        ## save csv for each batched result
        if save_each_chunk:
            output_filename = f"batch_{start_label:04d}_{end_label:04d}.csv"
            output_path = temp_dir / output_filename
            _save_output(chunk_result, output_path)

    if not results_chunked:
        print("error: no batched em3di result exist.")
    
    if results_chunked:
        df_concat_results = pd.concat(results_chunked, ignore_index = True)
        
        return df_concat_results
