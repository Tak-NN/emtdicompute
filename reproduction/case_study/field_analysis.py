import pandas as pd
import os

from emtdicompute.camera_handler import camera
from emtdicompute.scene import sample_mesh
from emtdicompute.model_analysis import field
from emtdicompute.batch import pipeline
import  emtdicompute.visualize_atframe as vis


FLOOR_OBJ_PATHS = ['./cs_floor_gf.obj', './cs_floor_1f.obj']
OBJ_PATH = './cs_building.obj'
DENSITY = 90
GRID = 1
OUTPUT_DIR = './outputs/field_analysis/'
OUTPUT_FNAME = 'field_analysis_gf.csv'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FNAME)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


for f in FLOOR_OBJ_PATHS:

    level = f[-6:-4]

    mesh = sample_mesh.mesh_sampling(OBJ_PATH, sampling_density=DENSITY, force_usecache=True)
    vpts = field.gen_grid_viewpoints(f, grid_size=GRID)

    field_emtdi_metrics = field.field_analysis_em3dimets(mesh, vpts, output_dir=OUTPUT_DIR, output_filename=OUTPUT_FNAME)
    field_emtdi_metrics.to_csv(OUTPUT_PATH, index = False)

    modes = ['vh', 'vj', 'vv', 'td']

    for m in modes:
        field.visualize_field_analysis(
            OUTPUT_PATH, 
            bldg_mesh_path=OBJ_PATH, 
            mode=m,
            camera_lookat=[16,0,-2.5],
            camera_front=[0,1,0],
            camera_up=[0,0,-1],
            save_img=True,
            img_output_path=f'./outputs/field_analysis/field_analysis_{level}_{m}.png',
            )
