import numpy as np
import torch
import inspect
import emtdicompute.utils.datatypes as dc
import pandas as pd
from emtdicompute.utils import debug_templates


# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def extract_camera_from_record(df_record: pd.DataFrame):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)
    """
    inputs:
        - df_record: pandas.DataFrame including at least position(x,y,z) and orientation(x,y,z) data
    output:
        - c_orig: camera origins based on coord_out
        - c_forward: camera forwards based on coord_out
    """

    origin_cols = {'Position_X', 'Position_Y', 'Position_Z'}
    forward_cols = {'HeadDir_X', 'HeadDir_Y', 'HeadDir_Z'}
    # timestamp_col = {'Timestamp'}

    n_of_rows = df_record.shape[0]

    if origin_cols.issubset(df_record):
        df_c_orig = df_record[["Position_X", "Position_Y", "Position_Z"]].copy()
    else:
        df_c_orig = pd.DataFrame(
            np.zeros((n_of_rows, 3)),
            columns = ['Position_X', 'Position_Y', 'Position_Z']
        )

    if forward_cols.issubset(df_record):
        df_c_forward = df_record[["HeadDir_X", "HeadDir_Y", "HeadDir_Z"]].copy()
    else:
        df_c_forward = pd.DataFrame(
            np.zeros((n_of_rows, 3)),
            columns = ['HeadDir_X', 'HeadDir_Y', 'HeadDir_Z']
        )

    # if timestamp_col.issubset(df_record):
    #     df_timestamp = df_record['Timestamp'].copy()
    # else:
    #     df_timestamp = pd.DataFrame(
    #         np.arange(n_of_rows),
    #         columns=['Timestamp']
    #     )
        
    cam_orig = df_c_orig.to_numpy()
    cam_forward = df_c_forward.to_numpy()
    # timestamp = df_timestamp.to_numpy()

    return cam_orig, cam_forward



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | 



def combine_camera_pos_and_dir(cam_orig, cam_forward) -> dc.CameraPosDir: 
    """
    inputs:
        - cam_orig: array of camera positions(x, y, z). ([F, 3] where F is total number of frames)
        - cam_forward: array of camera forwards(x, y, z). ([F, 3])
    outputs:
        - CameraPosDir@dataclass
    """

    # if timestamp is None:
        # timestamp = np.ndarray(len(cam_orig))
        
    return dc.CameraPosDir(cam_orig, cam_forward)



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | 



def camera_pyramids(combined_camera: dc.CameraPosDir, hfov=90, vfov=60, pyram_height=200) -> dc.PyramData:
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    ## -- extract inputs
    cam_orig = combined_camera.cam_orig
    cam_forward = combined_camera.cam_forward

    ### --- degree to rad
    hfov = hfov * np.pi / 180
    vfov = vfov * np.pi / 180

    ### --- set camera coordinate system
    world_up = ([0.0, 1.0, 0.0]) # y is up
    cam_forward = cam_forward / np.linalg.norm(cam_forward, axis=1, keepdims=True)
    cam_right= np.cross(cam_forward, world_up)
    cam_right = cam_right / np.linalg.norm(cam_right, axis=1, keepdims=True)
    cam_up = np.cross(cam_right, cam_forward)
    cam_up = cam_up / np.linalg.norm(cam_up, axis=1, keepdims=True)


    ### --- set camera pyramid vertices
    pyram_height = pyram_height
    pyram_v0 = cam_orig + cam_forward*pyram_height + cam_right*pyram_height*np.tan(hfov/2) + cam_up*pyram_height*np.tan(vfov/2)
    pyram_v1 = cam_orig + cam_forward*pyram_height - cam_right*pyram_height*np.tan(hfov/2) + cam_up*pyram_height*np.tan(vfov/2)
    pyram_v2 = cam_orig + cam_forward*pyram_height - cam_right*pyram_height*np.tan(hfov/2) - cam_up*pyram_height*np.tan(vfov/2)
    pyram_v3 = cam_orig + cam_forward*pyram_height + cam_right*pyram_height*np.tan(hfov/2) - cam_up*pyram_height*np.tan(vfov/2)

    pyram_points_allframe = np.stack([pyram_v0, pyram_v1, pyram_v2, pyram_v3, cam_orig], axis=1)
    print("points list:\n", pyram_points_allframe.shape)

    ### --- pyramid face normals
    def face_normal(p0, p1, p2):
        n = np.cross(p1 - p0, p2 - p0)
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        return n

    pyram_n0 = face_normal(cam_orig, pyram_v0, pyram_v3)
    pyram_n1 = face_normal(cam_orig, pyram_v2, pyram_v1)
    pyram_n2 = face_normal(cam_orig, pyram_v1, pyram_v0)
    pyram_n3 = face_normal(cam_orig, pyram_v3, pyram_v2)
    pyram_side_normals = np.stack([pyram_n0, pyram_n1, pyram_n2, pyram_n3])

    ### --- pyramid outline for o3d visualization
    pyram_lines = [
        [4, 0], [4, 1], [4, 2], [4, 3],
        [0, 1], [1, 2], [2, 3], [3, 0]
    ]

    ### --- output
    return dc.PyramData(pyram_points_allframe, np.array(pyram_lines), pyram_side_normals)



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | 



def pyramid_masking(
        mesh_sampling_outputs: dc.SampleMesh, 
        combined_camera: dc.CameraPosDir, 
        camera_pyramids_outputs: dc.PyramData, device='cuda'):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)
    """
    inputs:
        - mesh_sampling_outputs: SampleMesh@dataclass
        - load_camera_outputs: CameraPosDir@dataclass
        - camera_pyramids_outputs: PyramData@dataclass
    outputs:
        - pyramid_mask: true (false) for poitns inside (outside) pyramids. ([F, N])
    """

    ### --- extract inputs
    sampled_points = mesh_sampling_outputs.sampled_points
    cam_orig = combined_camera.cam_orig
    cam_forward = combined_camera.cam_forward
    pyram_side_normals = camera_pyramids_outputs.pyram_side_normals

    ### --- convert cpu to gpu
    pts = torch.as_tensor(sampled_points, dtype=torch.float32, device=device)
    c_orig = torch.as_tensor(cam_orig, dtype=torch.float32, device=device)
    c_forward = torch.as_tensor(cam_forward, dtype=torch.float32, device=device)
    p_normals = torch.as_tensor(pyram_side_normals, dtype=torch.float32, device=device)
    print("  - sampled_points from mesh_sampling():", pts.shape)
    print("  - c_orig:", c_orig.shape)
    print("  - pyramid_face_normals:", p_normals.shape)
    
    ### === exclude poitns out of camera pyramid === ###
    ## -- filter forward
    ray = pts.unsqueeze(0) - c_orig.unsqueeze(1)
    dots_forward = (ray * c_forward.unsqueeze(1)).sum(dim=-1)
    pyramid_mask = dots_forward > 0
    ## -- filter pyramid
    p_normals = p_normals.permute(1, 0, 2)
    dots_pyram = (ray.unsqueeze(2) * p_normals.unsqueeze(1)).sum(dim=-1)
    pyramid_mask = (dots_pyram > 0).all(dim=-1)
    
    return pyramid_mask