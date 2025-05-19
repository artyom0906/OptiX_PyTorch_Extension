#!/usr/bin/env python3
"""
OptiX VR Renderer - Python-driven implementation
"""

import sys
import os
import time
import math
import numpy as np
import torch
from sympy import false

# Add PyTorch library path to LD_LIBRARY_PATH to find libc10.so and other dependencies
try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    print(f"Adding PyTorch library path: {torch_lib_path}")
    
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_path
        
    # We need to restart the script for the environment change to take effect
    if not os.environ.get('OPENVR_TEST_RESTARTED'):
        print("Restarting script with updated LD_LIBRARY_PATH...")
        os.environ['OPENVR_TEST_RESTARTED'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)
except ImportError:
    print("Warning: PyTorch not found, continuing without library path adjustment")

# Import the extension
import optix_resource_system as ors
from pygame_monitor import PygameMonitor, get_gpu_info # Import the class and utility


def create_cube():
    """Create a simple cube geometry with texture coordinates"""
    # Using 24 vertices for 6 faces (4 vertices per face)
    vertices = torch.tensor([
        # Front face (z = -1)
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        # Back face (z = 1)
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        # Right face (x = 1)
        [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1],
        # Left face (x = -1)
        [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
        # Top face (y = 1)
        [-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1],
        # Bottom face (y = -1)
        [-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]
    ], dtype=torch.float32)
    
    # Each face has 2 triangles, with correct winding order
    indices = torch.tensor([
        # Front face
        [0, 3, 2], [0, 2, 1],
        # Back face
        [4, 5, 6], [4, 6, 7],
        # Right face
        [8, 11, 10], [8, 10, 9],
        # Left face
        [12, 13, 14], [12, 14, 15],
        # Top face
        [16, 17, 18], [16, 18, 19],
        # Bottom face
        [20, 23, 22], [20, 22, 21]
    ], dtype=torch.int32)
    
    # Each vertex gets its own texture coordinate
    texcoords = torch.tensor([
        # Front face
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        # Back face 
        [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        # Right face
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        # Left face
        [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        # Top face
        [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
        # Bottom face
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
    ], dtype=torch.float32)
    
    return vertices, indices, texcoords

def create_floor_plane(resource_manager, size=30.0, height=-2.0):
    """Create a simple floor plane geometry"""
    # Create a large quad for the floor
    half_size = size / 2.0
    vertices = torch.tensor([
        [-half_size, height, -half_size],  # 0: bottom-left
        [half_size, height, -half_size],   # 1: bottom-right
        [half_size, height, half_size],    # 2: top-right
        [-half_size, height, half_size]    # 3: top-left
    ], dtype=torch.float32)

    # Two triangles to form the quad
    indices = torch.tensor([
        [0, 2, 1], [0, 3, 2]  # counter-clockwise winding
    ], dtype=torch.int32)

    # Simple UV coordinates
    texcoords = torch.tensor([
        [0.0, 0.0],  # 0: bottom-left
        [10.0, 0.0], # 1: bottom-right
        [10.0, 10.0],# 2: top-right
        [0.0, 10.0]  # 3: top-left
    ], dtype=torch.float32)

    return vertices, indices, texcoords


def load_texture_from_file(resource_manager, file_path, flip_y=False):
    """Load a texture from file and create a texture resource"""
    try:
        from PIL import Image
        import numpy as np
        
        if not os.path.exists(file_path):
            print(f"Warning: Texture file not found: {file_path}")
            return None
            
        # Load and convert the image to RGBA
        image = Image.open(file_path).convert('RGBA')
        
        # Flip the image if needed
        if flip_y:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            
        # Convert to numpy array, then to torch tensor
        image_data = np.array(image).astype(np.float32) / 255.0  # Normalize to 0.0-1.0
        texture_tensor = torch.tensor(image_data, dtype=torch.float32)
        
        # Create texture resource
        texture_handle = resource_manager.create_texture(texture_tensor, ors.TextureType.RGBA)
        print(f"Loaded texture: {file_path}, handle: {texture_handle}")
        return texture_handle
        
    except Exception as e:
        print(f"Error loading texture {file_path}: {str(e)}")
        return None

def load_model_from_obj(resource_manager, obj_path, texture_params=None):
    """Load a model from an OBJ file with textures"""
    try:
        import trimesh
        
        if not os.path.exists(obj_path):
            print(f"Error: OBJ file not found: {obj_path}")
            return None
            
        print(f"Loading OBJ model: {obj_path}")
        
        # Load the mesh with trimesh
        mesh = trimesh.load(obj_path, force='mesh')
        
        # Extract vertex data
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        indices = torch.tensor(mesh.faces, dtype=torch.int32)
        
        # Extract texture coordinates if available
        tex_coords = None
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            tex_coords = torch.tensor(mesh.visual.uv, dtype=torch.float32)
            print(f"Model has texture coordinates: {tex_coords.shape}")
        
        # Extract vertex normals if available
        normals = None
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
            normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)
            print(f"Model has vertex normals: {normals.shape}")
        
        # Create geometry resource with all data
        geometry_params = {
            'vertices': vertices,
            'indices': indices
        }
        
        if tex_coords is not None:
            geometry_params['tex_coords'] = tex_coords
        
        if normals is not None:
            geometry_params['normals'] = normals
            
        geometry_handle = resource_manager.create_geometry(**geometry_params)
        
        # Load textures if provided
        texture_handles = {}
        if texture_params:
            for tex_type, tex_path in texture_params.items():
                if tex_path and os.path.exists(tex_path):
                    # Determine if we should flip textures based on type
                    # Usually normal maps should not be flipped in Y
                    flip_y = False#tex_type not in ["normalTexture", "albedoTexture"]
                    texture_handles[tex_type] = load_texture_from_file(
                        resource_manager, tex_path, flip_y=flip_y)
        return {
            "geometry": geometry_handle,
            "textures": texture_handles,
            "has_normals": normals is not None,
            "has_tex_coords": tex_coords is not None
        }
        
    except Exception as e:
        print(f"Error loading model {obj_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_scene(resource_manager):
    """Create and return a scene with renderable objects"""
    instances = []
    
    # Store gun instance to animate it
    global animated_gun_instance  # Will be used to animate the gun in the main loop
    animated_gun_instance = None
    
    # Create cube geometry
    vertices, indices, texcoords = create_cube()
    cube_geometry = resource_manager.create_geometry(
        vertices=vertices,
        indices=indices,
        tex_coords=texcoords
    )
    
    # Create a simple colored material
    cube_material = resource_manager.create_material(ors.LegacyMaterialType.LAMBERTIAN)
    resource_manager.set_material_parameter(cube_material, "albedo", [0.2, 0.4, 0.8])
    
    # Create cube instance
    cube_instance = ors.GeometryInstance(
        resource_manager,
        cube_geometry,
        cube_material
    )
    
    # Position the first cube in world space coordinates
    # In OpenVR/OptiX coordinate system: +X is right, +Y is up, +Z is forward
    cube_instance.set_transform(
        [0.0, 0.0, 6.0],                           # position fixed at (0,0,6) in front of origin
        [np.radians(0), np.radians(0), np.radians(0)], # no rotation
        [1.0, 1.0, 1.0]                               # normal scale
    )
    
    instances.append(cube_instance)
    
    # Create a checkered texture for the second cube
    texture_size = 128
    texture_data = torch.zeros((texture_size, texture_size, 4), dtype=torch.float32)
    
    # Create a checkerboard pattern with red and orange squares
    half = texture_size // 2
    square_size = texture_size // 8  # 8x8 grid of squares
    
    for i in range(texture_size):
        for j in range(texture_size):
            checkerboard_x = (i // square_size) % 2
            checkerboard_y = (j // square_size) % 2
            
            if (checkerboard_x + checkerboard_y) % 2 == 0:
                # Red squares
                texture_data[i, j, 0] = 0.9  # R
                texture_data[i, j, 1] = 0.1  # G
                texture_data[i, j, 2] = 0.1  # B
                texture_data[i, j, 3] = 1.0  # A
            else:
                # Orange squares
                texture_data[i, j, 0] = 0.9  # R
                texture_data[i, j, 1] = 0.5  # G
                texture_data[i, j, 2] = 0.1  # B
                texture_data[i, j, 3] = 1.0  # A
    
    # Create texture resource
    texture_handle = resource_manager.create_texture(texture_data, ors.TextureType.RGBA)
    
    # Try to load a gun model with PBR textures
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gun_model_path = os.path.join(base_dir, "models/gun/gun.obj")
    gun_textures = {
        "albedoTexture": os.path.join(base_dir, "models/gun/textures/famas-f1-atb-skin_baseColor.jpeg"),
        "normalTexture": os.path.join(base_dir, "models/gun/textures/famas-f1-atb-skin_normal.png"),
        "metallicRoughnessTexture": os.path.join(base_dir, "models/gun/textures/famas-f1-atb-skin_metallicRoughness.png")
    }
    
    gun_model = load_model_from_obj(resource_manager, gun_model_path, gun_textures)
    if gun_model:
        # Create a PBR material for the gun
        gun_material = resource_manager.create_material(ors.LegacyMaterialType.PBR)
        
        # Set material properties for PBR rendering
        resource_manager.set_material_parameter(gun_material, "albedo", [1.0, 1.0, 1.0])  # Base white color for texture
        resource_manager.set_material_parameter(gun_material, "metallic", 0.9)  # High metallic
        resource_manager.set_material_parameter(gun_material, "roughness", 0.3)  # Medium-low roughness
        
        # Set textures
        for tex_type, tex_handle in gun_model["textures"].items():
            resource_manager.set_material_parameter(gun_material, tex_type, tex_handle)
        
        # Create gun instance
        gun_instance = ors.GeometryInstance(
            resource_manager,
            gun_model["geometry"],
            gun_material
        )
        
        # Position the gun in the scene (position will be animated)
        gun_instance.set_transform(
            [-1.5, 2.0, 6.0],                         # position to the right of first cube
            [np.radians(0), np.radians(0), np.radians(0)],  # rotated to face camera
            [5, 5, 5]                          # larger scale to make it more visible
        )
        
        instances.append(gun_instance)
        print("Added gun model with PBR materials to the scene")
        
        # Store gun instance globally for animation
        #global animated_gun_instance
        animated_gun_instance = gun_instance
    
    # Create a material that uses the texture
    cube2_material = resource_manager.create_material(ors.LegacyMaterialType.LAMBERTIAN)
    # Base color white to show texture colors accurately
    resource_manager.set_material_parameter(cube2_material, "albedo", [1.0, 1.0, 1.0])
    # Set the texture
    resource_manager.set_material_parameter(cube2_material, "albedoTexture", texture_handle)
    
    # Create second cube instance using the same geometry
    cube2_instance = ors.GeometryInstance(
        resource_manager,
        cube_geometry,
        cube2_material
    )
    
    # Position the second cube to the left of the first cube
    cube2_instance.set_transform(
        [-3.0, 0.0, 6.0],                          # 3 units to the left of the first cube
        [np.radians(0), np.radians(0), np.radians(0)], # no rotation
        [1.0, 1.0, 1.0]                               # normal scale
    )
    
    instances.append(cube2_instance)
    
    # Create four additional cubes for the corners of the floor
    corner_positions = [
        [-8.0, 0.0, 12.0],  # Front-left corner
        [8.0, 0.0, 12.0],   # Front-right corner
        [-8.0, 0.0, 0.0],   # Back-left corner
        [8.0, 0.0, 0.0]     # Back-right corner
    ]
    
    # Create a different material for corner cubes
    corner_material = resource_manager.create_material(ors.LegacyMaterialType.LAMBERTIAN)
    resource_manager.set_material_parameter(corner_material, "albedo", [0.2, 0.7, 0.9])  # Light blue
    
    # Add a cube at each corner
    for i, pos in enumerate(corner_positions):
        corner_instance = ors.GeometryInstance(
            resource_manager,
            cube_geometry,
            corner_material
        )

        # Add some variation to each corner cube
        corner_instance.set_transform(
            pos,  # Position at corner
            [0, np.radians(45 * i), 0],  # Different rotation for each cube
            [0.8, 0.8, 0.8]  # Slightly smaller
        )

        instances.append(corner_instance)
    
    # 2. Create a green floor plane
    floor_vertices, floor_indices, floor_texcoords = create_floor_plane(resource_manager)
    floor_geometry = resource_manager.create_geometry(
        vertices=floor_vertices,
        indices=floor_indices,
        tex_coords=floor_texcoords
    )

    # Create a green material for the floor
    floor_material = resource_manager.create_material(ors.LegacyMaterialType.LAMBERTIAN)
    resource_manager.set_material_parameter(floor_material, "albedo", [0.2, 0.7, 0.3])  # Green color

    # Create floor instance
    floor_instance = ors.GeometryInstance(
        resource_manager,
        floor_geometry,
        floor_material
    )

    # Position the floor under and in front of the cube
    floor_instance.set_transform(
        [0.0, -4, 2.0],                             # position below cube
        [np.radians(0), np.radians(0), np.radians(0)], # no rotation
        [10.0, 10.0, 10.0]                                # normal scale
    )
    instances.append(floor_instance)

    return instances

def setup_vr_camera(
        eye_matrix,                  # <‑‑ now accepted but not used
        eye_to_head_matrix,          # 3×4 per‑eye
        head_pose_matrix,            # 3×4 HMD pose in world space
        proj_matrix,                 # 4×4 per‑eye projection
        base_pos=(0.0, 0.0, 0.0),    # locked‑camera fallback
        lock_position=False
):
    """
    Builds an ors.CameraParameters for OptiX ray generation (UVW method).

    Parameters
    ----------
    eye_matrix : 3×4
        Reserved for future use (time‑warp, etc.); ignored by this function.
    eye_to_head_matrix : 3×4
        Eye‑to‑head transform from OpenVR for this eye.
    head_pose_matrix : 3×4
        Headset pose in world space.
    proj_matrix : 4×4
        Asymmetric projection matrix for this eye.
    base_pos : 3‑tuple
        World‑space position used when lock_position is True.
    lock_position : bool
        If True, ignore positional tracking and keep the camera at base_pos.
    """
    #import numpy as np
    #from scipy.spatial.transform import Rotation as R
#
    #cam             = ors.CameraParameters()
    #head_pose_np    = np.asarray(head_pose_matrix, dtype=float).reshape(3, 4)
    #eye_to_head_np  = np.asarray(eye_to_head_matrix, dtype=float).reshape(3, 4)
    #proj_np         = np.asarray(proj_matrix,      dtype=float).reshape(4, 4)
#
    ## ---------- position ----------------------------------------------------
    #if lock_position:
    #    position = np.array(base_pos, dtype=np.float64)
    #else:
    #    eye_offset = eye_to_head_np[:, 3]
    #    hp4        = np.eye(4, dtype=np.float64)
    #    hp4[:3, :4] = head_pose_np
    #    position   = (hp4 @ np.append(eye_offset, 1.0))[:3]
    #    position[2] *= -1
#
    ## ---------- orientation (quaternion) -----------------------------------
    #rot_q = R.from_matrix(head_pose_np[:, :3])
    #yaw, pitch, roll = rot_q.as_euler("xyz", degrees=False)
#
    ## Uncomment any axis you want to invert
    #pitch = -pitch      # nodding
    #yaw   = -yaw        # turning
    #"# roll  = -roll       # tilting
#
    #rot_q   = R.from_euler("xyz", [yaw, pitch, roll])
    #right   = rot_q.apply([1, 0, 0])
    #up      = rot_q.apply([0, 1, 0])
    #forward = rot_q.apply([0, 0, -1])
#
    ## ---------- projection parameters --------------------------------------
    #tan_half_w = 1.0 / abs(proj_np[0, 0])
    #tan_half_h = 1.0 / abs(proj_np[1, 1])
    #off_x      = proj_np[0, 2]
    #off_y      = proj_np[1, 2]
#
    #camera_u = right   #* (2.0 * tan_half_w)
    #camera_v = up      #* (2.0 * tan_half_h)
    #camera_w = (position
    #            - tan_half_w * (off_x + 1.0) * right
    #            - tan_half_h * (off_y + 1.0) * up
    #            - forward)
#
    ## ---------- write to struct --------------------------------------------
    #cam.position  = position.tolist()
    #cam.camera_u  = camera_u.tolist()
    #cam.camera_v  = camera_v.tolist()
    #cam.camera_w  = camera_w.tolist()
    #return cam
    params = ors.CameraSetupParams()
    params.eye_matrix = eye_matrix
    params.eye_to_head_matrix = eye_to_head_matrix
    params.head_pose_matrix = head_pose_matrix
    params.proj_matrix = proj_matrix
    params.base_pos = base_pos
    params.lock_position = lock_position
    return ors.setup_vr_camera(params)

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OptiX VR Renderer')
    parser.add_argument('--lock-position', action='store_true', help='Lock camera position, only allow rotation')
    parser.add_argument('--position', type=float, nargs=3, default=[0.0, 1.7, 0.0], 
                        help='Fixed camera position (x, y, z) when locked, default: 0.0 1.7 0.0')
    parser.add_argument('--monitor_width', type=int, default=1920, help="Width of the Pygame monitor window")
    parser.add_argument('--monitor_height', type=int, default=1080, help="Height of the Pygame monitor window")

    args = parser.parse_args()
    
    print("=" * 80)
    print("OptiX VR Renderer (Python-driven)")
    print("=" * 80)
    print(f"Position lock: {'ON' if args.lock_position else 'OFF'}")
    if args.lock_position:
        print(f"Fixed position: {args.position}")
    
    # Create resource manager
    resource_manager = ors.ResourceManager()
    
    # Create scene objects
    instances = create_scene(resource_manager)
    
    # Create the OpenVR system with the resource manager and explicitly specify device 0
    vr_system = ors.OpenVRSystem(resource_manager, 0)

    # Create renderers for each eye
    left_renderer = ors.Renderer(resource_manager, 0)  # GPU 0
    right_renderer = ors.Renderer(resource_manager, 0) # Same GPU

    if not left_renderer.initialize() or not right_renderer.initialize():
        print("Failed to initialize renderers")
        vr_system.Shutdown()
        return

    # Add all instances to both renderers
    for instance in instances:
        left_renderer.add_instance(instance)
        right_renderer.add_instance(instance)

    monitor_instance = None
    try:
        # Initialize VR
        if not vr_system.Initialize():
            print("Failed to initialize OpenVR")
            return
            
        # Get recommended size
        width, height = vr_system.GetRecommendedRenderSize()
        print(f"Recommended render size: {width}x{height}")
            
        # Set renderers to VR system
        vr_system.SetRenderers(left_renderer, right_renderer)

        monitor_instance = PygameMonitor(
            vr_render_width_per_eye=width,
            vr_render_height_per_eye=height,
            monitor_window_width=args.monitor_width,
            monitor_window_height=args.monitor_height
        )
        monitor_instance.start()
        # Base position for the head
        head_pos = args.position
        last_fps_calc_time = time.perf_counter()
        frames_for_fps_calc = 0
        main_vr_fps = 0.0

        last_gpu_info_check_time = 0.0
        gpu_info_check_interval = 1.0
        current_system_gpu_stats = []
        print("- Press ESC in the desktop window to exit")
        
        # Main loop
        frame_count = 0
        print("Starting VR rendering loop...")

        loop_start_time = time.perf_counter()
        
        while monitor_instance.is_running and not vr_system.ShouldClose():
            loop_start_time = time.perf_counter()

            # Get all raw matrices from OpenVR
            left_eye_matrix, right_eye_matrix = vr_system.GetEyeTransforms()
            left_eye_to_head, right_eye_to_head = vr_system.GetEyeToHeadTransforms()
            head_pose_matrix = vr_system.GetHeadPoseMatrix()
            left_proj_matrix, right_proj_matrix = vr_system.GetProjectionMatrices()

            # Print head position occasionally for debug
            if frame_count % 300 == 0:
                head_matrix = np.array(head_pose_matrix).reshape(3, 4)
                print(f"Head position: {head_matrix[:, 3]}")
                
            # Animate the gun if it exists
            if animated_gun_instance:
                # Get current time for smooth animation
                current_time = time.time()
                
                # Calculate rotation angle based on time
                rotation_speed = 0.3  # radians per second (increased for faster rotation)
                rotation_angle = current_time * rotation_speed % (2 * np.pi)
                
                # Calculate vertical floating motion
                float_speed = 0.4  # oscillations per second
                float_height = 0.15  # maximum height change in meters (increased)
                vertical_offset = float_height * np.sin(current_time * float_speed * 2 * np.pi)
                
                # Add a slight bobbing in X and Z directions
                horizontal_speed = 0.2  # oscillations per second
                horizontal_amount = 0.1  # maximum horizontal movement
                x_offset = horizontal_amount * np.sin(current_time * horizontal_speed * 2 * np.pi)
                z_offset = horizontal_amount * np.cos(current_time * horizontal_speed * 2 * np.pi)
                
                # Use a fixed scale
                current_scale = [5, 5, 5]  # Slightly larger
                
                # Update position with floating effect
                position = [
                    -1.5 + x_offset,  # X position with slight movement
                    2.0 + vertical_offset,  # Y position with floating effect
                    6.0 + z_offset  # Z position with slight movement
                ]
                
                # Update rotation around Y axis with slight tilting
                rotation = [
                    np.radians(5) * np.sin(current_time * 0.3),  # X rotation (slight tilting)
                    rotation_angle + np.radians(180),  # Y rotation with animation + initial 180° rotation
                    np.radians(5) * np.cos(current_time * 0.3)   # Z rotation (slight tilting)
                ]
                
                # Apply the updated transform
                animated_gun_instance.set_transform(position, rotation, current_scale)
                right_renderer.set_scene_changed()
                left_renderer.set_scene_changed()

            # Update camera settings using all available matrix information
            left_camera = setup_vr_camera(
                left_eye_matrix, 
                left_eye_to_head, 
                head_pose_matrix, 
                left_proj_matrix, 
                head_pos,
                args.lock_position
            )
            
            right_camera = setup_vr_camera(
                right_eye_matrix, 
                right_eye_to_head, 
                head_pose_matrix, 
                right_proj_matrix, 
                head_pos,
                args.lock_position
            )
            
            # Print head position occasionally
            #if frame_count % 300 == 0:
            #    head_matrix = np.array(head_pose_matrix).reshape(3, 4)
            #    print(f"Head position: {head_matrix[:, 3]}")
            
            # Make sure the camera positions are properly set
            left_renderer.set_camera(left_camera)
            right_renderer.set_camera(right_camera)
            
            # Render frame to headset
            vr_system.RenderFrame()


            # --- Gather Data for Pygame Monitor ---
            data_for_monitor = {}
            data_for_monitor["left_eye_cpu_tensor"] = vr_system.get_last_left_eye_cpu()
            data_for_monitor["right_eye_cpu_tensor"] = vr_system.get_last_right_eye_cpu()

            data_for_monitor["left_render_ms"] = vr_system.get_left_eye_render_time_ms()
            data_for_monitor["left_optix_copy_ms"] = vr_system.get_left_eye_internal_copy_time_ms()
            data_for_monitor["left_texture_copy_ms"] = vr_system.get_left_eye_texture_copy_time_ms()
            data_for_monitor["left_gpu_to_cpu_ms"] = vr_system.get_left_eye_to_cpu_copy_time_ms()
            data_for_monitor["left_cpu_to_gpu_ms"] = vr_system.get_left_eye_from_cpu_copy_time_ms()

            data_for_monitor["right_render_ms"] = vr_system.get_right_eye_render_time_ms()
            data_for_monitor["right_optix_copy_ms"] = vr_system.get_right_eye_internal_copy_time_ms()
            data_for_monitor["right_texture_copy_ms"] = vr_system.get_right_eye_texture_copy_time_ms()
            data_for_monitor["right_gpu_to_cpu_ms"] = vr_system.get_right_eye_to_cpu_copy_time_ms()
            data_for_monitor["right_cpu_to_gpu_ms"] = vr_system.get_right_eye_from_cpu_copy_time_ms()

            data_for_monitor["total_to_cpu_ms"] = vr_system.get_total_to_cpu_copy_time_ms()
            data_for_monitor["total_from_cpu_ms"] = vr_system.get_total_from_cpu_copy_time_ms()

            frames_for_fps_calc += 1
            current_time_fps = time.perf_counter()
            if current_time_fps - last_fps_calc_time >= 1.0:
                main_vr_fps = frames_for_fps_calc / (current_time_fps - last_fps_calc_time)
                frames_for_fps_calc = 0
                last_fps_calc_time = current_time_fps
            data_for_monitor["vr_fps"] = main_vr_fps

            if current_time_fps - last_gpu_info_check_time > gpu_info_check_interval:
                current_system_gpu_stats = get_gpu_info()
                last_gpu_info_check_time = current_time_fps
            data_for_monitor["system_gpu_stats"] = current_system_gpu_stats

            loop_end_time = time.perf_counter()
            data_for_monitor["main_loop_time_ms"] = (loop_end_time - loop_start_time) * 1000.0

            monitor_instance.update_data(data_for_monitor)


            # Process events
            vr_system.PollEvents()
            
            # Print occasional update
            frame_count += 1
            if frame_count % 500 == 0:
                print(f"Rendered {frame_count} frames")
            
            # Sleep to limit CPU usage
            #time.sleep(1/90)  # Target 90 FPS
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up
        vr_system.Shutdown()
        print("VR system shutdown complete")

if __name__ == "__main__":
    main()