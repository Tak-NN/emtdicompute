import pandas as pd
import os

from emtdicompute.camera_handler import camera
from emtdicompute.scene import sample_mesh
from emtdicompute.batch import pipeline
import emtdicompute.visualize_atframe as vis
from emtdicompute.model_analysis import field


OBJ_PATH = './cs_building.obj'
TRAJ_PATHS = ['./cam_path_dir_A.csv', './cam_path_dir_B.csv']
DENSITY = 90
CACHE_DIR = './mesh_cache/'

output_dir = './outputs/path_analysis'

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

mesh = sample_mesh.mesh_sampling(OBJ_PATH, sampling_density = DENSITY, force_usecache=True)

# limited FOV
for traj in TRAJ_PATHS:

    df_traj = pd.read_csv(traj)
    c_orig, c_fwd = camera.extract_camera_from_record(df_traj)
    vantage_pts = camera.combine_camera_pos_and_dir(c_orig, c_fwd)

    df_output = pipeline.compute_em3di_metrics(mesh, vantage_pts)
    
    scenario = os.path.basename(traj)[4:-4]
    print(scenario)
    filename = f'scenario_{scenario}.csv'
    df_output.to_csv(os.path.join(output_dir, filename), index = False)


# unlimited FOV
traj = TRAJ_PATHS[0]
df_traj = pd.read_csv(traj)
c_orig, c_fwd = camera.extract_camera_from_record(df_traj)
vantage_pts = camera.combine_camera_pos_and_dir(c_orig, c_fwd)

filename = 'scenario_unlimfov.csv'

field.field_analysis_em3dimets(mesh, vantage_pts, output_dir=output_dir, output_filename=filename)

