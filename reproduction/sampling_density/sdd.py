import pandas as pd
import os 
import random
import numpy as np

from emtdicompute.model_analysis import field
from emtdicompute.scene import sample_mesh
from emtdicompute.camera_handler import camera
from pathlib import Path
import emtdicompute.visualize_atframe as vis
import objname_setting as objst

obj_name = objst.OBJ_FILENAME
obj_basename = obj_name[:-4]
vpt_name = objst.VPT_FILENAME

dict_mesh_vp = {
    obj_name : vpt_name,
    }

densities = np.arange(5, 200, 10)
print(densities)

seeds = pd.read_csv('./seeds.csv').to_numpy()
seeds = np.squeeze(seeds)
## uncomment below and modify the number of iteration.
# seeds = seeds[:20]

for s in seeds:
    cache_dir = f"./outputs/{obj_basename}_mesh/s_{s}"

    for d in densities:

        for key in dict_mesh_vp.keys():
            obj_path = key
            cam_path = dict_mesh_vp[key]
            print(f'{obj_path}, {cam_path}')

            df_vp = pd.read_csv(cam_path)
            c_orig, c_fwd = camera.extract_camera_from_record(df_vp)
            vp = camera.combine_camera_pos_and_dir(c_orig, c_fwd)

            mesh = sample_mesh.mesh_sampling(
                obj_path, 
                sampling_density = d, 
                seed = s,
                force_usecache=False, 
                force_recompute=True,
                cache_dir = Path(cache_dir)
                )
            
            print(obj_basename)

            output_filename = f"{obj_basename}_{d}_{s}_field.csv"
            
            field.field_analysis_em3dimets(mesh,vp,output_dir = f'./outputs/{obj_basename}/',output_filename=output_filename)