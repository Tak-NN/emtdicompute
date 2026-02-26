import pandas as pd
from pathlib import Path
from emtdicompute.camera_handler import camera
from emtdicompute.scene import sample_mesh
from emtdicompute.model_analysis import field
from emtdicompute.batch import pipeline
import emtdicompute.visualize_atframe as vis




DICT_MESH_VP = {
    "model_01_cube.obj": "cam_01.csv",
    "model_02_tallcube.obj": "cam_02.csv",
    "model_03_widecube.obj": "cam_03.csv",
    }

DENSITY = 50
CACHE_DIR = './outputs/mesh_sampled'

for key in DICT_MESH_VP.keys():
    obj_path = key
    cam_path = DICT_MESH_VP[key]
    print(f"{obj_path}, {cam_path}")

    df_vp = pd.read_csv(cam_path)
    c_orig, c_fwd = camera.extract_camera_from_record(df_vp)
    vp = camera.combine_camera_pos_and_dir(c_orig, c_fwd)

    mesh = sample_mesh.mesh_sampling(
        obj_path,
        sampling_density=DENSITY,
        cache_dir = Path(CACHE_DIR),
        force_recompute=True,
        force_usecache=False,
    )

    obj_name = obj_path[12:-4]

    output_filename = f"{obj_name}_validation_d{DENSITY}.csv"

    field.field_analysis_em3dimets(
        mesh,
        vp,
        output_dir='./outputs',
        output_filename=output_filename
    )
