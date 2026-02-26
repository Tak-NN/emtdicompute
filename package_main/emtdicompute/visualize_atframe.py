from emtdicompute.em3di_visualization import visualizer


def visualize(
        record_path: str = None,
        mesh_path: str = None,
        frame: int = None,
        density: float = None,
        mode: str = None, # select from 'vh' or 'td'
        width = 960,
        height = 720,
        background_color = None,
        color = None, # mesh line color
        camera_lookat = None,
        camera_front = None,
        camera_up = None,
        camera_zoom = None,
        save_img = True,
        img_output_path = None,
):
    if record_path is None:
        record_path = str(input('[enter record filepath]'))
    if mesh_path is None:
        mesh_path = str(input('[enter mesh filepath]'))
    if frame is None:
        frame = int(input('[enter frame number]'))
    if density is None:
        density = float(input('[enter density]'))

    geoms_vhfaces_atframe = visualizer.visualize_visible_vhfaces_atframe(record_path, mesh_path, frame = frame, density = density, mode = mode)
    geoms_bldg = visualizer.visualize_bldg(mesh_path, color=color)
    geoms = geoms_vhfaces_atframe + geoms_bldg
    
    visualizer.open_visualization_window(
        geoms,
        width=width,
        height=height,
        background_color=background_color,
        camera_lookat=camera_lookat,
        camera_front=camera_front,
        camera_up=camera_up,
        camera_zoom=camera_zoom,
        )
    
    if save_img:
        if img_output_path is None:
            img_output_path = '.visible_points.png'
        
        visualizer.save_visualization_image(
            geoms,
            img_output_path,
            width=width,
            height=height,
            background_color=background_color,
            camera_lookat = camera_lookat,
            camera_front = camera_front,
            camera_up = camera_up,
            camera_zoom = camera_zoom,
        )

if __name__ == '__main__':
    visualize()