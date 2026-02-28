import pandas as pd
import os
from pathlib import Path

from emtdicompute.camera_handler import camera
from emtdicompute.scene import sample_mesh
from emtdicompute.batch import pipeline
import  emtdicompute.visualize_atframe as vis

OBJ_PATH = './casestudy_building.obj'
DICT_VP_PATH = {
    # 'dirA':'./scenario_point_analysis_dirA.csv',
    'dirA':'./path_dir_A.csv',
    # 'dirB':'./scenario_point_analysis_dirB.csv',
    }
DENSITY = 90
CACHE_DIR = './mesh_cache'



for d in DICT_VP_PATH.keys():
    vp = DICT_VP_PATH[d]
    print(d, vp)


    vis.visualize(
        vp, OBJ_PATH, mode = 'td',
        density=DENSITY, frame = 20,
        background_color=[1,1,1], color=[0,0,0],
        camera_lookat=[21,0,-15],
        camera_front = [0,0.5,1],
        camera_up=[0,1,-0.5],
        camera_zoom = .7, 
        img_output_path=f"./outputs/point_analyses/visipts_{d}_A_.png"
        )
    
