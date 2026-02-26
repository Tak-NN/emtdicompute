import torch
import numpy as np
import trimesh
import inspect
from emtdicompute.utils import debug_templates
import emtdicompute.utils.datatypes as dc
import sys



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def embodied_3d_isovist_prelims(
        mesh_sampling_outputs: dc.SampleMesh, 
        combined_camera: dc.CameraPosDir, 
        pyramid_mask: torch.Tensor,
        device='cuda') -> dc.Em3diPrelims:
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)
    """
    outputs:
      - dc.Em3diPrelims: 
        # necessary information to compute em3di metrics (e.g., vh_ratio, td_ratio)
          - visible_points_allframes: points(x, y, z) directly visible from camera. ([N_v, 3] where N_v is the total number of directly visible points in all frames.)
          - vhlbls_visipts_allframes: vertical / horizontal labels for each visible_points_allframes. ([N_v, ])
          - normals_visipts_allframes: normal vectors of each visible_points_allframes. ([N_v, 3])
          - frame_indices: indices representing the frames to which the points belong. ([N_v, ])
    """

    ### --- extract inputs
    sampled_points = mesh_sampling_outputs.sampled_points
    face_vh_labels = mesh_sampling_outputs.face_vh_labels
    face_normals = mesh_sampling_outputs.face_normals
    mesh = mesh_sampling_outputs.mesh
    cam_orig = combined_camera.cam_orig
    cam_forward = combined_camera.cam_forward

    ### --- data conversion
    if torch.is_tensor(sampled_points):
        pts = sampled_points
    else:
        pts = torch.as_tensor(sampled_points, dtype=torch.float32, device=device)
    if torch.is_tensor(face_vh_labels):
        vhlbls = face_vh_labels
    else:
        vhlbls = torch.as_tensor(face_vh_labels, dtype=torch.int32, device=device)
    if torch.is_tensor(face_normals):
        normals = face_normals
    else:
        normals = torch.as_tensor(face_normals, dtype=torch.float32, device=device)
    c_orig = torch.as_tensor(cam_orig, dtype=torch.float32, device=device)
    c_forward = torch.as_tensor(cam_forward, dtype=torch.float32, device=device)
    print('sampled_points from mesh_sampling(): ', pts.shape)

    ### --- visibility mask tensor
    pyramid_mask = pyramid_mask.bool().to(device)
    F, N = pyramid_mask.shape
    print("pytamid mask shape: ", pyramid_mask.shape)

    ### === ray-mesh intersection === ###
    ## -- trimesh
    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    ## -- buffer
    visible_points_allframes = []
    vhlbls_visipts_allframes = []
    normals_visipts_allframes = []
    frame_indices = []

    print("  -------------------------------------------")
    print("  -------------------------------------------")

    ### --- main
    for f in range(F):
        print("frame (in batch): ", f)
        mask_f = pyramid_mask[f]
        if not mask_f.any():
            print(" No points in the pyramid.")
            print("-------------------------------------------")
            continue

        pts_f = pts[mask_f]
        lbls_f = vhlbls[mask_f]
        normls_f = normals[mask_f]
        cam_pos = c_orig[f]
        print("  -pts_f:", pts_f.shape)

        ray_dirs = pts_f - cam_pos
        ray_dirs = ray_dirs / (ray_dirs.norm(dim=1, keepdim=True) + 1e-8)
        cam_pos_np = cam_pos.cpu().numpy()
        ray_dirs_np = ray_dirs.cpu().numpy()
        pts_f_np = pts_f.cpu().numpy()
        lbls_f_np = lbls_f.cpu().numpy()
        normals_f_np = normls_f.cpu().numpy()

        ### --- ray-mesh intersection
        ## -- set hit_points shape to avoid error
        hit_points = np.full((pts_f_np.shape[0], 3), np.nan)
        ### --- hit detection
        print("  -default hit points:", hit_points.shape)
        _, ray_id, _hit_points = ray_intersector.intersects_id(
            ray_origins=cam_pos_np[None, :].repeat(ray_dirs_np.shape[0], axis=0),
            ray_directions=ray_dirs_np,
            multiple_hits=False,
            return_locations=True
        )
        hit_points[ray_id] = _hit_points

        print("  -n of intersection (ray id):", ray_id.shape)
        print("  -hit points:", hit_points.shape)

        ### --- visibility check
        target_dist = np.linalg.norm(pts_f_np - cam_pos_np, axis=1)
        hit_dist = np.linalg.norm(hit_points - cam_pos_np, axis=1)
        visible_indices = np.logical_or(np.isnan(hit_dist), hit_dist > target_dist - 1e-4) # true for visible points
        print("  - n of visible points:", visible_indices.sum())

        ### --- send to gpu
        if np.any(visible_indices):
            visible_points_allframes.append(torch.tensor(pts_f_np[visible_indices], device=device))
            vhlbls_visipts_allframes.append(torch.tensor(lbls_f_np[visible_indices], device=device))
            normals_visipts_allframes.append(torch.tensor(normals_f_np[visible_indices], device=device))
            frame_indices.append(
                torch.full((visible_indices.sum(),), f, device=device, dtype=torch.long)
            )

        debug_templates.comment_section1()

    if len(visible_points_allframes) == 0:
        empty_points = torch.empty((0, 3), device=device)
        empty_labels = torch.empty((0,), device=device, dtype=torch.int32)
        empty_normals = torch.empty((0, 3), device=device)
        empty_indices = torch.empty((0,), device=device, dtype=torch.long)
        return dc.Em3diPrelims(empty_points, empty_labels, empty_normals, empty_indices, total_frames=F)

    visible_points_allframes = torch.cat(visible_points_allframes, dim=0)
    vhlbls_visipts_allframes = torch.cat(vhlbls_visipts_allframes, dim=0)
    normals_visipts_allframes = torch.cat(normals_visipts_allframes, dim=0)
    frame_indices = torch.cat(frame_indices, dim=0)

    print("####===###=== isovist visibile points summary ===###===####")
    print("  - visible points:", visible_points_allframes.shape)
    print("  - vh labels for visible points:", vhlbls_visipts_allframes.shape)
    print("  - normals for visible points:", normals_visipts_allframes.shape)
    print("  - frame indices:", frame_indices.shape)
    print("####===###===###===###===============###===###===###===####")

    return dc.Em3diPrelims(
        visible_points_allframes,
        vhlbls_visipts_allframes,
        normals_visipts_allframes,
        frame_indices,
        total_frames=F,
    )



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | 



def em3disovist_vh_ratio(em3disovist_outputs: dc.Em3diPrelims, combined_camera: dc.CameraPosDir, device='cuda'):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    ## -- extract inputs
    visible_points_allframes = em3disovist_outputs.visible_points_allframes
    vhlbls_visipts_allframes = em3disovist_outputs.vhlbls_visipts_allframes
    normals_visipts_allframes = em3disovist_outputs.normals_visipts_allframes
    frame_indices = em3disovist_outputs.frame_indices.to(device=device, dtype=torch.long)
    total_frames = em3disovist_outputs.total_frames
    c_orig = torch.as_tensor(combined_camera.cam_orig, dtype=torch.float32, device=device)
    print("  - visible points:", visible_points_allframes.shape)
    print("  - vh labels for visible points:", vhlbls_visipts_allframes.shape)
    print("  - normals for visible points:", normals_visipts_allframes.shape)
    print("  - frame indices:", frame_indices.shape)

    if total_frames == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)

    ### --- seperation of horizontal / vertical visible (intersection) points
    # - h
    mask_h = (vhlbls_visipts_allframes == 0)
    visipts_h = visible_points_allframes[mask_h]
    normals_h = normals_visipts_allframes[mask_h]
    frame_indices_h = frame_indices[mask_h]
    c_orig_h = c_orig[frame_indices_h]
    #- v
    mask_v = (vhlbls_visipts_allframes == 1)
    visipts_v = visible_points_allframes[mask_v]
    normals_v = normals_visipts_allframes[mask_v]
    frame_indices_v = frame_indices[mask_v]
    c_orig_v = c_orig[frame_indices_v]

    ### --- height of pyramid
    heights_h = _height_of_pyramid(c_orig_h, visipts_h, normals_h)
    heights_v = _height_of_pyramid(c_orig_v, visipts_v, normals_v)

    ## -- prepare buffer
    sum_h = torch.zeros(total_frames, dtype=torch.float32, device=device)
    sum_v = torch.zeros(total_frames, dtype=torch.float32, device=device)
    cnt_per_frame_h = torch.zeros(total_frames, dtype=torch.float32, device=device)
    cnt_per_frame_v = torch.zeros(total_frames, dtype=torch.float32, device=device)
    ## -- accumulate heights for each frame
    # - h
    if frame_indices_h.numel() > 0:
        sum_h.scatter_add_(0, frame_indices_h, heights_h)
        cnt_visipts_h = torch.ones_like(heights_h, dtype=torch.float32, device=device)
        cnt_per_frame_h.scatter_add_(0, frame_indices_h, cnt_visipts_h)
    mean_heights_h = sum_h / (cnt_per_frame_h + 1e-8)
    debug_templates.comment_section1()
    print("Vertical pyarmid heights (Bottom faces are horizontal):")
    print("  - sum_height_of_pyramid_per_frame:", sum_h.shape)
    print("  - num_of_ray-Hface_intersect_per_frame:", cnt_per_frame_h)
    print("  - mean_height:", mean_heights_h)

    # - v
    if frame_indices_v.numel() > 0:
        sum_v.scatter_add_(0, frame_indices_v, heights_v)
        cnt_visipts_v = torch.ones_like(heights_v, dtype=torch.float32, device=device)
        cnt_per_frame_v.scatter_add_(0, frame_indices_v, cnt_visipts_v)
    mean_heights_v = sum_v / (cnt_per_frame_v + 1e-8)
    print("Horizontal pyramid heights (Bottom faces are vertical):")
    print("  - sum_height_of_pyramid_per_frame:", sum_v.shape)
    print("  - num_of_ray-Hface_intersect_per_frame:", cnt_per_frame_v)
    print("  - mean_height:", mean_heights_v)

    ### --- vh ratio
    vh_ratio = torch.full((total_frames,), -2.0, dtype=torch.float32, device=device)
    valid_frames = (cnt_per_frame_h > 0) & (cnt_per_frame_v > 0)
    vh_ratio[valid_frames] = mean_heights_h[valid_frames] / (mean_heights_v[valid_frames] + 1e-8)
    debug_templates.comment_section1()
    print("VH ratio:", vh_ratio)
    if frame_indices.numel() > 0:
        print("HV ratio size:\n", vh_ratio.shape, "==frame_num_check==>", (frame_indices.max()+1 == vh_ratio.shape[0]).item())

    return vh_ratio



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | 



def em3disovist_td_ratio(em3disovist_outputs: dc.Em3diPrelims, combined_camera: dc.CameraPosDir, device='cuda'):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    ## -- extract inputs
    visible_points_allframes = em3disovist_outputs.visible_points_allframes
    vhlbls_visipts_allframes = em3disovist_outputs.vhlbls_visipts_allframes
    normals_visipts_allframes = em3disovist_outputs.normals_visipts_allframes
    frame_indices = em3disovist_outputs.frame_indices.to(device=device, dtype=torch.long)
    total_frames = em3disovist_outputs.total_frames
    c_orig = torch.as_tensor(combined_camera.cam_orig, dtype=torch.float32, device=device)
    print("  - visible points:", visible_points_allframes.shape)
    print("  - vh labels for visible points:", vhlbls_visipts_allframes.shape)
    print("  - normals for visible points:", normals_visipts_allframes.shape)
    print("  - frame indices:", frame_indices.shape)

    if total_frames == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)

    ### --- seperation of top(ceiling) / down(floor) intersection points
    mask_h = (vhlbls_visipts_allframes == 0) # horizontal faces
    # - t
    mask_t = mask_h & (normals_visipts_allframes[:,1] < 0)
    visipts_t = visible_points_allframes[mask_t]
    normals_t = normals_visipts_allframes[mask_t]
    frame_indices_t = frame_indices[mask_t]
    c_orig_t = c_orig[frame_indices_t]
    # - d
    mask_d = mask_h & (normals_visipts_allframes[:,1] > 0)
    visipts_d = visible_points_allframes[mask_d]
    normals_d = normals_visipts_allframes[mask_d]
    frame_indices_d = frame_indices[mask_d]
    c_orig_d = c_orig[frame_indices_d]

    ### --- height of pyramid
    heights_t = _height_of_pyramid(c_orig_t, visipts_t, normals_t)
    heights_d = _height_of_pyramid(c_orig_d, visipts_d, normals_d)
    print("points on top faces:", visipts_t.shape)
    print("heights t:", heights_t.shape)

    ### --- td ratio
    ## -- buffer
    sum_t = torch.zeros(total_frames, dtype=torch.float32, device=device)
    sum_d = torch.zeros(total_frames, dtype=torch.float32, device=device)

    ## -- accumulate heights for each frame
    # - t
    if frame_indices_t.numel() > 0:
        sum_t.scatter_add_(0, frame_indices_t, heights_t)
        print(f"SUM_T: {sum_t}")    
    # - d
    if frame_indices_d.numel() > 0:
        sum_d.scatter_add_(0, frame_indices_d, heights_d)
    print("pyramid volume (D):", sum_d)

    ### --- td ratio
    td_ratio = torch.full((total_frames,), -2.0, dtype=torch.float32, device=device)
    valid_frames = (sum_t > 0) & (sum_d > 0)
    td_ratio[valid_frames] = sum_t[valid_frames] / (sum_d[valid_frames] + 1e-8)
    print("TD ratio:", td_ratio)

    return td_ratio



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | 



def em3disovist_v_jaggedness(em3disovist_outputs: dc.Em3diPrelims, combined_camera: dc.CameraPosDir, sampled_mesh: dc.SampleMesh, device = 'cuda'):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)

    ## -- extract inputs
    visible_points_allframes = em3disovist_outputs.visible_points_allframes
    vhlbls_visipts_allframes = em3disovist_outputs.vhlbls_visipts_allframes
    normals_visipts_allframes = em3disovist_outputs.normals_visipts_allframes
    frame_indices = em3disovist_outputs.frame_indices.to(device=device, dtype=torch.long)
    total_frames = em3disovist_outputs.total_frames
    c_orig = torch.as_tensor(combined_camera.cam_orig, dtype=torch.float32, device=device)

    if total_frames == 0:
        return torch.empty((0,), dtype = torch.float32, device = device)
    
    mask_h = (vhlbls_visipts_allframes == 0) # bottom face of vertical pyramid is horizontal face
    visipts_h = visible_points_allframes[mask_h]
    normals_h = normals_visipts_allframes[mask_h]
    frame_indices_h = frame_indices[mask_h]
    c_orig_h = c_orig[frame_indices_h]

    vert_pyramid_heights = _height_of_pyramid(c_orig_h, visipts_h, normals_h)

    sum_v_vol = torch.zeros(total_frames, dtype = torch.float32, device = device)
    sum_v_area = torch.zeros(total_frames, dtype = torch.float32, device = device)

    if frame_indices_h.numel() > 0:
        sum_v_vol.scatter_add_(0, frame_indices_h, vert_pyramid_heights / (sampled_mesh.density * 3))
        cnt_visipts_v = torch.ones_like(vert_pyramid_heights, dtype=torch.float32, device=device)
        sum_v_area.scatter_add_(0, frame_indices_h, cnt_visipts_v / sampled_mesh.density)
    
    ### --- vertical jaggedness
    v_jaggedness = torch.full((total_frames, ), -2.0, dtype = torch.float32, device = device)
    valid_frames = (sum_v_area > 0) & (sum_v_vol > 0)
    v_jaggedness[valid_frames] = (sum_v_vol[valid_frames] ** (1/3)) / (sum_v_area[valid_frames] ** (1/2) + 1e-8)
    print("Vertical Jaggedness:", v_jaggedness)

    return v_jaggedness



# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



def em3disovist_visible_volume(
        em3disovist_outputs: dc.Em3diPrelims,
        combined_camera: dc.CameraPosDir,
        sampled_mesh: dc.SampleMesh,
        device = 'cuda',
):
    debug_templates.comment_section_func(inspect.currentframe().f_code.co_name)


    ## -- extract inputs
    visible_points_allframes = em3disovist_outputs.visible_points_allframes
    normals_visipts_allframes = em3disovist_outputs.normals_visipts_allframes
    frame_indices = em3disovist_outputs.frame_indices.to(device=device, dtype=torch.long)
    total_frames = em3disovist_outputs.total_frames
    c_orig = torch.as_tensor(combined_camera.cam_orig, dtype=torch.float32, device=device)

    if total_frames == 0:
        return torch.empty((0,), dtype = torch.float32, device = device)
    
    pyramid_heights = _height_of_pyramid(c_orig[frame_indices], visible_points_allframes, normals_visipts_allframes)

    sum_heights = torch.zeros(total_frames, dtype = torch.float32, device = device)

    if frame_indices.numel() > 0:
        sum_heights.scatter_add_(0, frame_indices, pyramid_heights)

    visible_volume = torch.full((total_frames, ), -2.0, dtype = torch.float32, device = device)
    valid_frames = sum_heights >= 0
    visible_volume[valid_frames] = sum_heights[valid_frames] * (1/3) / sampled_mesh.density

    return visible_volume
    

    
# | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # | # |



## == "heigh of pyramid" by Krukar et al. (2021) == ##
def _height_of_pyramid(_top, _intersection, _normal):
    v = _intersection - _top
    height = torch.abs((v * _normal).sum(dim=1))
    return height
