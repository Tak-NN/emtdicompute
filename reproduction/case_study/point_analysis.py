import pandas as pd
import os
from pathlib import Path

from emtdicompute.camera_handler import camera
from emtdicompute.scene import sample_mesh
from emtdicompute.batch import pipeline
import  emtdicompute.visualize_atframe as vis
from emtdicompute.model_analysis import field

OBJ_PATH = './cs_building.obj'
VANTAGE_PATHS = [
    './cam_point_analysis_dirA.csv',
    './cam_point_analysis_dirB.csv',
]
DENSITY = 90
CACHE_DIR = './mesh_cache/'

output_path = './outputs/point_analysis'

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

mesh = sample_mesh.mesh_sampling(OBJ_PATH, sampling_density=DENSITY, force_usecache=True)

for v in VANTAGE_PATHS:
    scenario = v[-8:-4]
    print(scenario)
    df_vpt = pd.read_csv(v)
    c_orig, c_fwd = camera.extract_camera_from_record(df_vpt)
    vpt = camera.combine_camera_pos_and_dir(c_orig, c_fwd)

    # with the limited FOV
    df_output = pipeline.compute_em3di_metrics(mesh, vpt)
    df_output.to_csv(f'{output_path}/cs_point_analysis_{scenario}.csv', index = False)


# with the unlimited FOV
field.field_analysis_em3dimets(
    mesh, 
    vpt,
    output_dir=output_path, 
    output_filename='cs_point_analysis_unlimfov.csv',
    )


# uncomment below to obtain the visualisation of visible sampled points from the vantage point.
# vis.visualize(
#     record_path = VANTAGE_PATHS[0], # 0 = dirA, 1 = dirB
#     mesh_path = OBJ_PATH,
#     mode = 'vh', # vh or td
#     density=DENSITY,
#     save_img=False, # True to save image
#     img_output_path='./outputs/point_analysis/visipts.png',
#     frame = 0, # do not change
#     )
