#!/usr/bin/env python3
"""
OptiX Resource System - Multi-GPU Renderer

This script creates a simple cube and renders it using multiple GPUs,
each with a different camera angle, then combines the images.
Includes real-time GPU monitoring for performance tracking.
"""

import math
import time
import numpy as np
import torch
import pygame
import sys
import os
import subprocess
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from matplotlib import pyplot as plt

# Add parent directory to path to import Python modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import resource system modules
from python.controllers.KeyboardController import KeyboardController
from python.models.Camera import Camera
from python.models.Player import Player
import optix_resource_system as ors

# Set thread mode for PyTorch to ensure thread safety
torch.set_num_threads(1)  # Limit internal PyTorch threading to avoid conflicts
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads

# GPU monitoring functions
def get_gpu_info():
    """Get GPU information using nvidia-smi command"""
    try:
        # Run nvidia-smi to get GPU stats
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu', 
             '--format=csv,noheader,nounits'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse output
        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            # Parse CSV data
            values = [v.strip() for v in line.split(',')]
            if len(values) >= 7:
                gpu_stats.append({
                    'id': int(values[0]),
                    'name': values[1],
                    'gpu_util': float(values[2]),
                    'memory_util': float(values[3]),
                    'memory_used': float(values[4]),
                    'memory_total': float(values[5]),
                    'temperature': float(values[6])
                })
        
        return gpu_stats
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

# Initialize pygame
pygame.init()

# Create a window - doubled width to fit two images side by side
width, height = 800, 600
full_width = width * 2 + 10  # Add 10px border between images
window = pygame.display.set_mode((full_width, height))
pygame.display.set_caption("OptiX Multi-GPU Renderer")

def create_floor():
    """Create a floor plane geometry with texture coordinates"""
    # Floor vertices - large plane on XZ plane (Y=0)
    size = 10.0  # Large floor size
    vertices = torch.tensor([
        [-size, 0.0, -size],  # Bottom left
        [size, 0.0, -size],   # Bottom right
        [size, 0.0, size],    # Top right
        [-size, 0.0, size]    # Top left
    ], dtype=torch.float32)
    
    # Floor indices (2 triangles) - make sure they face upward (Y+)
    indices = torch.tensor([
        [0, 1, 2],  # First triangle - counter-clockwise winding
        [0, 2, 3]   # Second triangle - counter-clockwise winding
    ], dtype=torch.int32)
    
    # Add texture coordinates for the floor - repeat texture multiple times
    texcoords = torch.tensor([
        [0.0, 0.0],  # Bottom left
        [5.0, 0.0],  # Bottom right - repeat texture 5 times
        [5.0, 5.0],  # Top right - repeat texture 5x5 times
        [0.0, 5.0]   # Top left - repeat texture 5 times
    ], dtype=torch.float32)
    
    print("Created floor plane with upward-facing normal and texture coordinates")
    return vertices, indices, texcoords

def create_cube():
    """Create a simple cube geometry with texture coordinates"""
    # We need a different set of vertices for proper texture mapping - indexed cube
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
    
    # Each face has 2 triangles, with correct winding order for outward-facing normals
    indices = torch.tensor([
        # Front face - CCW winding from outside 
        [0, 3, 2], [0, 2, 1],
        # Back face - CCW winding from outside
        [4, 5, 6], [4, 6, 7],
        # Right face - CCW winding from outside
        [8, 11, 10], [8, 10, 9],
        # Left face - CCW winding from outside
        [12, 13, 14], [12, 14, 15],
        # Top face - CCW winding from outside
        [16, 17, 18], [16, 18, 19],
        # Bottom face - CCW winding from outside
        [20, 23, 22], [20, 22, 21]
    ], dtype=torch.int32)
    
    # Each vertex gets its own texture coordinate
    # Using the standard layout: each face gets (0,0) -> (1,1) mapping
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
    
    print("Created cube with proper per-face texture coordinates")
    return vertices, indices, texcoords

def create_renderer_with_scene(resource_manager, device_id):
    """Create a renderer with a cube and floor scene on the specified device"""
    try:
        # Import OBJ loader
        from obj_loader import ObjLoader

        obj_loader = ObjLoader(resource_manager)

        # Try to load a teapot model from OBJ file if it exists
        # Use full path for model files
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        teapot_result = obj_loader.load_obj(
            os.path.join(base_dir, "models/teapot/teapot.obj"),
            os.path.join(base_dir, "models/teapot/default.png")
        )
    except Exception as e:
        print(f"Unable to load OBJ model: {e}")
        print("Continuing with basic scene")

    # Create cube geometry with texture coordinates
    cube_vertices, cube_indices, cube_texcoords = create_cube()
    cube_geometry = resource_manager.create_geometry(
        vertices=cube_vertices, 
        indices=cube_indices, 
        tex_coords=cube_texcoords
    )

    # Create a much simpler texture with clear red, green, blue, and white quadrants
    # This will make it easier to diagnose texture mapping issues
    texture_size = 128  # Smaller texture for faster loading
    texture_data = torch.zeros((texture_size, texture_size, 4), dtype=torch.float32)
    
    half = texture_size // 2
    
    # Top-left quadrant: Pure red
    texture_data[0:half, 0:half, 0] = 1.0  # R
    texture_data[0:half, 0:half, 3] = 1.0  # A
    
    # Top-right quadrant: Pure green
    texture_data[0:half, half:, 1] = 1.0  # G
    texture_data[0:half, half:, 3] = 1.0  # A
    
    # Bottom-left quadrant: Pure blue
    texture_data[half:, 0:half, 2] = 1.0  # B
    texture_data[half:, 0:half, 3] = 1.0  # A
    
    # Bottom-right quadrant: White
    texture_data[half:, half:, 0] = 1.0  # R
    texture_data[half:, half:, 1] = 1.0  # G
    texture_data[half:, half:, 2] = 1.0  # B
    texture_data[half:, half:, 3] = 1.0  # A
    
    # Add small central marker for orientation
    center = texture_size // 2
    marker_size = texture_size // 16
    
    # Black square in the center
    texture_data[center-marker_size:center+marker_size, 
                 center-marker_size:center+marker_size, :] = 0.0
    
    # Small yellow cross in the center 
    cross_size = marker_size // 2
    texture_data[center-cross_size:center+cross_size, center-1:center+1, 0] = 1.0  # R
    texture_data[center-cross_size:center+cross_size, center-1:center+1, 1] = 1.0  # G
    texture_data[center-1:center+1, center-cross_size:center+cross_size, 0] = 1.0  # R
    texture_data[center-1:center+1, center-cross_size:center+cross_size, 1] = 1.0  # G

    filename="texture.png"
    plt.figure(figsize=(10, 10))
    plt.imshow(texture_data)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Image saved to {filename}")

    # Debug the texture data shape before passing to create_texture
    print(f"Texture data shape: {texture_data.shape}")
    print(f"Texture data type: {texture_data.dtype}")
    print(f"Texture data min/max values: {texture_data.min().item()}, {texture_data.max().item()}")
    
    # Check the memory layout of the tensor - should be contiguous
    print(f"Texture data is contiguous: {texture_data.is_contiguous()}")
    
    # Ensure the tensor is in the correct memory layout (order)
    if not texture_data.is_contiguous():
        texture_data = texture_data.contiguous()
    
    # When we call create_texture, the data will be:
    # - Height x Width x 4 channels (RGBA)
    # - Float32 data type
    # - Values between 0.0 and 1.0
    # - Contiguous memory layout
    texture_handle = resource_manager.create_texture(texture_data, ors.TextureType.RGBA)
    
    # Create a blue material for the cube
    cube_material = resource_manager.create_material(ors.LegacyMaterialType.LAMBERTIAN)
    resource_manager.set_material_parameter(cube_material, "albedo", [0.2, 0.4, 0.8])
    resource_manager.set_material_parameter(cube_material, "albedoTexture", texture_handle)

    # Create floor geometry
    floor_vertices, floor_indices, _ = create_floor()
    floor_geometry = resource_manager.create_geometry(floor_vertices, floor_indices)

    # Create a green material for the floor
    floor_material = resource_manager.create_material(ors.LegacyMaterialType.LAMBERTIAN)
    resource_manager.set_material_parameter(floor_material, "albedo", [0.2, 0.8, 0.2])

    # List to store all instances
    instances = []

    # Create cube instance
    cube_instance = ors.GeometryInstance(
        resource_manager,
        cube_geometry,
        cube_material
    )
    instances.append(cube_instance)

    # Place the cube in front of the camera
    cube_instance.set_transform(
        [0.0, 0.5, -6.0],                              # position at z=-6
        [np.radians(0), np.radians(0), np.radians(0)], # no rotation
        [1.0, 1.0, 1.0]                                # standard scale
    )

    #cube_instance1 = ors.GeometryInstance(
    #    resource_manager,
    #    cube_geometry,
    #    cube_material
    #)
    #instances.append(cube_instance1)

    # Place the cube in front of the camera
    #cube_instance1.set_transform(
    #    [4.0, 0.5, -6.0],                              # position at z=-6
    #    [np.radians(0), np.radians(0), np.radians(0)], # no rotation
    #    [1.0, 1.0, 1.0]                                # standard scale
    #)
    # Try to load OBJ models if available
    #try:
        # Initialize OBJ loader


    cube_material1 = resource_manager.create_material(ors.LegacyMaterialType.LAMBERTIAN)
    resource_manager.set_material_parameter(cube_material1, "albedo", [0.2, 0.4, 0.8])
    resource_manager.set_material_parameter(cube_material1, "albedoTexture", texture_handle)
        #print("!!!TEAPOT", teapot_result[0])
    cube1_geometry = resource_manager.create_geometry(
        vertices=teapot_result[0],
        indices=teapot_result[1],
        tex_coords=teapot_result[2],
    )

    cube_instance1 = ors.GeometryInstance(
        resource_manager,
        cube1_geometry,
        cube_material1
    )
    instances.append(cube_instance1)

        # Place the cube in front of the camera
    cube_instance1.set_transform(
        [-2.0, -2, -6.0],                              # position at z=-6
        [np.radians(0), np.radians(0), np.radians(0)], # no rotation
        [0.02, 0.02, 0.02]                                # standard scale
    )
        #
        #if "instance" in teapot_result:
        #    teapot_instance = teapot_result["instance"]
        #    instances.append(teapot_instance)
        #    teapot_instance.set_transform(
        #        [-4.0, 0.0, -6.0],                          # position left of cube
        #        [np.radians(0), np.radians(0), np.radians(0)], # slight rotation
        #        [0.02, 0.02, 0.02]                            # half size
        #    )
        #    print("Added teapot model to scene")
    #except Exception as e:
    #    print(f"Unable to load OBJ model: {e}")
    #    print("Continuing with basic scene")

    # Create floor instance
    floor_instance = ors.GeometryInstance(
        resource_manager,
        floor_geometry,
        floor_material
    )
    instances.append(floor_instance)

    ## Place the floor under the cube
    floor_instance.set_transform(
        [0.0, -1.0, -6.0],                             # position below cube
        [np.radians(0), np.radians(0), np.radians(0)], # no rotation
        [6.0, 0.1, 6.0]                                # wide but thin
    )



    print(f"Creating renderer on device {device_id}...")
    renderer = ors.Renderer(resource_manager, 0)
    renderer1 = ors.Renderer(resource_manager, 1)
    # Log that we're initializing with explicit device ID
    print(f"Initializing OptiX pipeline on GPU {device_id}...")
    if not renderer.initialize():
        print(f"Failed to initialize renderer on device {device_id}")
        return None
    if not renderer1.initialize():
        print(f"Failed to initialize renderer on device {device_id}")
        return None
    print(f"Successfully initialized OptiX pipeline on GPU {device_id}")
    
    # Add all instances to both renderers
    for instance in instances:
        renderer.add_instance(instance)
        renderer1.add_instance(instance)
    
    return renderer, renderer1

def main():
    """Main function"""
    print("Starting OptiX Multi-GPU Renderer with Thread Affinity...")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Warning: Only {num_gpus} GPU(s) available. Continuing with available device(s).")
    else:
        print(f"Found {num_gpus} GPUs.")
        
    # Print GPU information
    for i in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1024**2:.0f} MB)")
        
    # Set thread affinity mode if available
    try:
        if hasattr(torch.cuda, 'set_device') and hasattr(torch, 'multiprocessing'):
            torch.multiprocessing.set_sharing_strategy('file_system')
            print("Using file_system sharing strategy for thread safety")
    except Exception as e:
        print(f"Warning: Could not configure thread sharing: {e}")
    
    # Create resource manager
    print("Creating resource manager...")
    resource_manager = ors.ResourceManager()

    # Create renderers for each GPU (or at least 2 if available)
    renderers = list(create_renderer_with_scene(resource_manager, 0))
    
    if len(renderers) < 1:
        print("Failed to create any renderers. Exiting.")
        return
    
    # Set up camera and controls
    controller = KeyboardController()
    print("Using keyboard controls")
    
    # Create camera for first view (centered)
    # Using negative Y as up vector since we'll flip the image
    device = torch.device("cuda:0")
    camera1 = Camera(
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),  # Position higher up at the origin
        torch.tensor([0.0, 0.5, -6.0], dtype=torch.float32), # Look at the cube position
        torch.tensor([0, 1, 0], dtype=torch.float32)        # Negative Y up vector (image will be flipped)
    )
    
    # Create camera for second view (shifted to the left by 0.5 on X axis)
    camera2 = Camera(
        torch.tensor([-0.5, 1.0, 0.0], dtype=torch.float32), # Offset to the left of first camera
        torch.tensor([0.0, 0.5, -6.0], dtype=torch.float32), # Look at the same cube position
        torch.tensor([0, 1, 0], dtype=torch.float32)        # Negative Y up vector (image will be flipped)
    )
    
    # Create player for interactive control (controls first camera)
    player = Player(controller, [camera1, camera2])
    
    # Print controls explanation
    print("\nKEYBOARD CONTROLS:")
    print("  W/S        - Move forward/backward")
    print("  A/D        - Move left/right")
    print("  Q/E        - Move up/down")
    print("  Arrow keys - Look around")
    print("  ESC        - Exit program\n")
    
    # Main render loop
    running = True
    clock = pygame.time.Clock()
    frame_times = [0.0]
    frame_count = 0  # Track frame count for optimizations
    
    # Setup for GPU monitoring
    gpu_stats = []
    last_gpu_check = 0
    gpu_check_interval = 1.0  # Check GPU stats every 1 second
    
    # Store stats for later analysis
    monitoring_data = []
    
    # Create a larger window to accommodate GPU stats
    window_height = height + 150  # Add space for GPU stats at bottom
    window = pygame.display.set_mode((full_width, window_height))
    
    print("Entering main loop...")
    while running:
        # Record the start time for FPS calculation
        start_time = time.perf_counter()
        
        # Get GPU stats periodically
        current_time = time.time()
        if current_time - last_gpu_check >= gpu_check_interval:
            gpu_stats = get_gpu_info()
            last_gpu_check = current_time
            
            # Record stats with timestamp
            monitoring_data.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'gpu_stats': gpu_stats,
                'fps': 1.0 / frame_times[-1] if len(frame_times) > 1 and frame_times[-1] > 0 else 0
            })
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update player position/camera for the first view
        player.update(frame_times[-1])
        
        # Update second camera - follow first camera but stay offset to the left
        #camera2.position = camera1.position.clone()
        #camera2.position[0] = camera1.position[0] - 0.5  # Stay 0.5 units to the left
        #camera2.lookat = camera1.lookat.clone()  # Look at the same point
        
        # Set up camera parameters for both renderers
        #camera_params1 = ors.CameraParameters()
        #camera_params1.position = camera1.position.tolist()
        #camera_params1.look_at = camera1.lookat.tolist()
        #camera_params1.up = camera1.up.tolist()
        #camera_params1.fov = 90.0  # Much wider field of view to be sure we see the cube
        #camera_params1.aspect_ratio = width / height
        #
        #camera_params2 = ors.CameraParameters()
        #camera_params2.position = camera2.position.tolist()
        #camera_params2.look_at = camera2.lookat.tolist()
        #camera_params2.up = camera2.up.tolist()
        #camera_params2.fov = 90.0  # Much wider field of view to be sure we see the cube
        #camera_params2.aspect_ratio = width / height

        # Get camera properties from your Python Camera object
        py_camera_position = camera1.position.numpy() # Ensure it's a NumPy array
        py_camera_lookat = camera1.lookat.numpy()
        py_camera_up = camera1.up.numpy()
        fov_degrees = 90.0
        aspect_ratio = float(width) / height

        # Calculate camera basis vectors (u, v, w)
        # This logic is similar to how you might set up a view matrix

        # W (forward, but OptiX uses view direction from eye, so often -forward)
        # OptiX shaders often expect W to be the vector from eye to the center of the view plane's bottom-left pixel.
        # For simplicity, let's derive U,V,W in world space for direct raygen first.

        # Z-axis of camera (forward direction)
        z_axis = py_camera_lookat - py_camera_position
        z_axis = z_axis / np.linalg.norm(z_axis)

        # X-axis of camera (right vector)
        x_axis = np.cross(py_camera_up, z_axis) # Note: OpenVR/OpenGL might use (forward x up)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y-axis of camera (up vector)
        y_axis = np.cross(z_axis, x_axis)
        # y_axis = py_camera_up / np.linalg.norm(py_camera_up) # Or re-normalize the provided up

        # Calculate w_vec (direction to center of view plane, not just forward)
        # This depends on your ray generation shader. The current Renderer.cu
        # expects camera_pos, camera_u, camera_v, camera_w.
        # camera_w is typically: eye - (u_scale * u) - (v_scale * v) - forward_vec
        # For a simple perspective projection:
        fov_rad = math.radians(fov_degrees)
        h = math.tan(fov_rad / 2.0)

        # These are scaled basis vectors for the view plane
        # The raygen shader then uses: ray_dir = normalize(params.camera_u * screen_x + params.camera_v * screen_y + params.camera_w)
        # Where params.camera_w is effectively the direction to the center of the near plane.
        # The 'w' in CameraParameters is more like the direction to the bottom-left of the view plane center.
        # Let's follow the Renderer.cu LaunchParams expectations where:
        # params.camera_w is the vector to the center of the image plane from the origin of the camera's coordinate system.
        # And camera_u, camera_v are scaled basis vectors.

        # Let's assume the C++ CameraParameters u,v,w are the basis vectors scaled by FOV/aspect.
        # This needs to match exactly how `LaunchParams` in Renderer.cu uses them.
        # From Renderer.cu:
        # launchParams.camera_u = m_camera.u;
        # launchParams.camera_v = m_camera.v;
        # launchParams.camera_w = m_camera.w; // This 'w' is the forward vector for the view.
        # The actual ray direction in shader_minimal.cu:
        # dir = params.camera_u * screen_pos.x + params.camera_v * screen_pos.y + params.camera_w

        # So camera_w should be the view direction (normalized forward)
        # And camera_u, camera_v should be scaled versions of the right and up vectors.

        u_vector = x_axis * (aspect_ratio * h) # Scaled right vector
        v_vector = y_axis * h                  # Scaled up vector
        w_vector = z_axis                      # Forward vector (view direction)

        # For camera 1
        camera_params1 = ors.CameraParameters()
        camera_params1.position = py_camera_position.tolist()
        camera_params1.camera_u = u_vector.tolist()
        camera_params1.camera_v = v_vector.tolist()
        camera_params1.camera_w = w_vector.tolist()

        # For camera 2 (you'll need to recalculate u_vector, v_vector, w_vector for camera2 as well)
        py_camera_position2 = camera2.position.numpy()
        py_camera_lookat2 = camera2.lookat.numpy()
        py_camera_up2 = camera2.up.numpy()

        z_axis2 = py_camera_lookat2 - py_camera_position2
        z_axis2 = z_axis2 / np.linalg.norm(z_axis2)
        x_axis2 = np.cross(py_camera_up2, z_axis2)
        x_axis2 = x_axis2 / np.linalg.norm(x_axis2)
        y_axis2 = np.cross(z_axis2, x_axis2)

        u_vector2 = x_axis2 * (aspect_ratio * h)
        v_vector2 = y_axis2 * h
        w_vector2 = z_axis2

        camera_params2 = ors.CameraParameters()
        camera_params2.position = py_camera_position2.tolist()
        camera_params2.camera_u = u_vector2.tolist()
        camera_params2.camera_v = v_vector2.tolist()
        camera_params2.camera_w = w_vector2.tolist()


        # Set cameras for renderers
        if len(renderers) > 0:
            renderers[0].set_camera(camera_params1)
        if len(renderers) > 1:
            renderers[1].set_camera(camera_params2)
        
        # Render the scene on both GPUs using separate threads
        images = [None] * len(renderers)
        
        def render_on_gpu(idx, renderer, first_frame=False):
            """Render on a specific GPU using a dedicated thread with explicit GPU binding"""
            gpu_id = 0 if idx == 0 else (1 if num_gpus > 1 else 0)
            try:
                # Explicitly set CUDA device for this thread
                torch.cuda.set_device(gpu_id)
                
                # Measure pure rendering time (GPU computation only)
                start_render_time = time.perf_counter()
                
                # On the first frame, build the acceleration structure
                # On subsequent frames, just update camera and skip rebuilding
                output_image = renderer.render(width, height, first_frame)
                
                # Record rendering time (without CPU transfer)
                pure_render_time = time.perf_counter() - start_render_time
                
                # Now measure the CPU transfer time separately
                start_copy_time = time.perf_counter()
                
                # This is the expensive operation we've optimized in the C++ code
                # Now the tensor only gets moved to CPU here, not in the renderer
                result = output_image.cpu().numpy()
                
                # Calculate copy time
                copy_time = time.perf_counter() - start_copy_time
                
                # Total time includes both rendering and copying
                render_time = pure_render_time + copy_time
                
                # Log timing information
                #print(f"GPU {gpu_id}: render={pure_render_time*1000:.1f}ms, copy={copy_time*1000:.1f}ms, total={render_time*1000:.1f}ms")
                
                # Store detailed timing information for display
                if not hasattr(render_on_gpu, 'timing_details'):
                    render_on_gpu.timing_details = {}
                
                render_on_gpu.timing_details[gpu_id] = {
                    'pure_render_time': pure_render_time,
                    'copy_time': copy_time,
                    'total_time': render_time
                }
                
                return result, render_time, gpu_id
            except Exception as e:
                print(f"Error rendering on GPU {gpu_id}: {e}")
                return np.zeros((height, width, 3), dtype=np.float32), 0.0, gpu_id
        
        # Store render times for each GPU
        render_times = {}
        
        # Keep track of whether this is the first frame for optimization
        first_frame = frame_count == 0
        
        # Use ThreadPoolExecutor to run renderers in parallel
        with ThreadPoolExecutor(max_workers=len(renderers)) as executor:
            # Start all rendering tasks
            futures = []
            for i, renderer in enumerate(renderers):
                futures.append(executor.submit(render_on_gpu, i, renderer, first_frame))
            
            # Wait for all to complete and collect results
            for i, future in enumerate(futures):
                result, render_time, gpu_id = future.result()
                images[i] = result
                render_times[gpu_id] = render_time
        
        # Increment frame counter
        frame_count += 1
        
        # Create a combined image with a border between the two views
        if len(images) == 2:
            # Create a blank full-width image
            full_image = np.zeros((height, full_width, 3), dtype=np.float32)
            
            # Flip images vertically (make Y+ point upward)
            flipped_images = [img for img in images]#np.flip(img, axis=0)
            
            # Copy first image to the left side (flipped)
            full_image[:, 0:width, :] = flipped_images[0]
            
            # Draw a vertical border in the middle (gray)
            full_image[:, width:width+10, :] = 0.5
            
            # Copy second image to the right side (flipped)
            full_image[:, width+10:, :] = flipped_images[1]
            
            # For PyGame, ensure proper scaling and RGB format
            pygame_image = np.transpose(full_image, (1, 0, 2))  # PyGame expects height, width, channels
        else:
            # Use just one image if only one renderer is available
            flipped_image = np.flip(images[0], axis=0)  # Flip vertically
            pygame_image = np.transpose(flipped_image, (1, 0, 2))
        
        # Scale to 0-255 uint8 for PyGame
        pygame_image = (pygame_image * 255).astype(np.uint8)
        
        # Create PyGame surface
        surface = pygame.surfarray.make_surface(pygame_image[:, :, :3])
        
        # Display the rendered image
        window.blit(surface, (0, 0))
        
        # Display instructions
        font = pygame.font.Font(None, 24)
        
        # Display camera info
        pos_rounded = [round(x, 2) for x in camera1.position.tolist()]
        cam1_text = font.render(f"Camera 1 (left): {pos_rounded}", True, (255, 255, 255))
        window.blit(cam1_text, (10, 40))
        
        pos_rounded2 = [round(x, 2) for x in camera2.position.tolist()]
        cam2_text = font.render(f"Camera 2 (right): {pos_rounded2}", True, (255, 255, 255))
        window.blit(cam2_text, (10, 70))
        
        # Store additional timing information
        render_details = getattr(render_on_gpu, 'timing_details', {})
        
        # Add view labels with detailed timing information
        # Left view (GPU 0)
        left_label = "Left View"
        if 0 in render_times:
            total_ms = render_times[0]*1000
            # Get render and copy times if available
            pure_render_ms = render_details.get(0, {}).get('pure_render_time', 0) * 1000
            copy_ms = render_details.get(0, {}).get('copy_time', 0) * 1000
            
            # Add timing breakdown
            if pure_render_ms > 0 or copy_ms > 0:
                left_label = f"GPU 0: Render={pure_render_ms:.1f}ms, Copy={copy_ms:.1f}ms"
            else:
                left_label += f" - {total_ms:.1f}ms"
                
        left_text = font.render(left_label, True, (255, 255, 255))
        window.blit(left_text, (width/2 - 140, 10))
        
        # Right view (GPU 1)
        right_label = "Right View"
        if 1 in render_times:
            total_ms = render_times[1]*1000
            # Get render and copy times if available
            pure_render_ms = render_details.get(1, {}).get('pure_render_time', 0) * 1000
            copy_ms = render_details.get(1, {}).get('copy_time', 0) * 1000
            
            # Add timing breakdown
            if pure_render_ms > 0 or copy_ms > 0:
                right_label = f"GPU 1: Render={pure_render_ms:.1f}ms, Copy={copy_ms:.1f}ms, Total={pure_render_ms+copy_ms:.1f}ms"
            else:
                right_label += f" - {total_ms:.1f}ms"
                
        right_text = font.render(right_label, True, (255, 255, 255))
        window.blit(right_text, (width + width/2 - 140, 10))
        

        
        # Draw GPU statistics at the bottom
        if gpu_stats:
            # Draw a dark gray background for stats
            stats_rect = pygame.Rect(0, height, full_width, 150)
            pygame.draw.rect(window, (60, 60, 60), stats_rect)
            
            # Draw border line
            pygame.draw.line(window, (120, 120, 120), (0, height), (full_width, height), 2)
            
            # Stats header
            font_header = pygame.font.Font(None, 28)
            header = font_header.render("GPU MONITORING", True, (240, 240, 240))
            window.blit(header, (20, height + 10))
            
            # Draw each GPU's stats
            font_stats = pygame.font.Font(None, 24)
            for i, gpu in enumerate(gpu_stats):
                # Calculate vertical offset for each GPU (30px separation)
                y_offset = height + 40 + (i * 50)
                
                # Determine color based on GPU usage (green < 50%, yellow 50-80%, red > 80%)
                util_color = (0, 255, 0) if gpu['gpu_util'] < 50 else (255, 255, 0) if gpu['gpu_util'] < 80 else (255, 0, 0)
                mem_color = (0, 255, 0) if gpu['memory_util'] < 50 else (255, 255, 0) if gpu['memory_util'] < 80 else (255, 0, 0)
                
                # GPU name and ID
                gpu_title = font_stats.render(f"GPU {gpu['id']} ({gpu['name']})", True, (255, 255, 255))
                window.blit(gpu_title, (20, y_offset))
                
                # Set consistent positions for all GPUs
                label_x = 300     # X position for labels
                value_x = 380     # X position for value text (percentages) 
                bar_x = 580       # X position for bars - moved further right to accommodate 32GB text
                bar_width = 250   # Width of bars
                
                # GPU utilization with colored bar
                util_text = font_stats.render("GPU:", True, (255, 255, 255))
                window.blit(util_text, (label_x, y_offset))
                
                util_value = font_stats.render(f"{gpu['gpu_util']:.1f}%", True, (255, 255, 255))
                window.blit(util_value, (value_x, y_offset))
                
                # Draw utilization bar
                bar_rect = pygame.Rect(bar_x, y_offset, bar_width, 20)
                pygame.draw.rect(window, (100, 100, 100), bar_rect)  # Background
                fill_width = int((gpu['gpu_util'] / 100) * bar_width)
                fill_rect = pygame.Rect(bar_x, y_offset, fill_width, 20)
                pygame.draw.rect(window, util_color, fill_rect)  # Filled portion
                
                # Memory usage on next line
                mem_label = font_stats.render("Memory:", True, (255, 255, 255))
                window.blit(mem_label, (label_x, y_offset + 25))
                
                mem_text = font_stats.render(f"{gpu['memory_used']:.0f}MB/{gpu['memory_total']:.0f}MB ({gpu['memory_util']:.1f}%)", True, (255, 255, 255))
                window.blit(mem_text, (value_x, y_offset + 25))
                
                # Draw memory bar (aligned with GPU utilization bar)
                mem_bar_rect = pygame.Rect(bar_x, y_offset + 25, bar_width, 20)
                pygame.draw.rect(window, (100, 100, 100), mem_bar_rect)  # Background
                mem_fill_width = int((gpu['memory_util'] / 100) * bar_width)
                mem_fill_rect = pygame.Rect(bar_x, y_offset + 25, mem_fill_width, 20)
                pygame.draw.rect(window, mem_color, mem_fill_rect)  # Filled portion
                
                # Temperature (aligned with other elements)
                temp_label_x = bar_x + bar_width + 30  # Position after the bar
                temp_color = (0, 255, 0) if gpu['temperature'] < 70 else (255, 255, 0) if gpu['temperature'] < 80 else (255, 0, 0)
                temp_text = font_stats.render(f"Temp: {gpu['temperature']:.0f}°C", True, temp_color)
                window.blit(temp_text, (temp_label_x, y_offset))
                
                # Add small colored indicator next to temperature
                indicator_size = 15
                indicator_x = temp_label_x + 120
                indicator_rect = pygame.Rect(indicator_x, y_offset + 3, indicator_size, indicator_size)
                pygame.draw.rect(window, temp_color, indicator_rect)

                # Display render time difference if both GPUs are used
        if 0 in render_times and 1 in render_times:
            time_diff = abs(render_times[0] - render_times[1]) * 1000  # Convert to ms
            diff_pct = (max(render_times[0], render_times[1]) / min(render_times[0], render_times[1]) - 1.0) * 100 if min(render_times[0], render_times[1]) > 0 else 0

            # Color based on difference percentage
            diff_color = (255, 255, 100)  # Yellow by default
            if diff_pct > 30:
                diff_color = (255, 100, 100)  # Red for large difference
            elif diff_pct < 10:
                diff_color = (100, 255, 100)  # Green for small difference

            # Render time difference with percentage
            diff_text = font.render(f"Render time diff: {time_diff:.1f}ms ({diff_pct:.1f}%)", True, diff_color)
            window.blit(diff_text, (full_width - 290, height + 70))

            # Show which GPU is faster
            faster_gpu = 0 if render_times[0] < render_times[1] else 1
            faster_text = font.render(f"GPU {faster_gpu} is faster", True, diff_color)
            window.blit(faster_text, (full_width - 290, height + 40))
        # Update display
        pygame.display.flip()
        
        # Calculate frame time
        end_time = time.perf_counter()
        frame_time = end_time - start_time
        frame_times.append(frame_time)
        
        # Update FPS in window title every 10 frames
        if len(frame_times) % 10 == 0:
            avg_frame_time = sum(frame_times[-10:]) / 10.0
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            pygame.display.set_caption(f"OptiX Multi-GPU Renderer - FPS: {fps:.2f}")
            
            # Display current FPS in window with better positioning
            fps_text = font_stats.render(f"FPS: {fps:.1f}", True, (255, 255, 100))
            fps_rect = fps_text.get_rect()
            fps_rect.topright = (full_width - 30, height + 15)  # Position in top-right of stats area
            window.blit(fps_text, fps_rect.topleft)
        
        # Limit to 60 FPS
        clock.tick(60)
    
    # Clean up
    pygame.quit()
    print("Application closed")
    
    # Save monitoring data to file
    if len(monitoring_data) > 0:
        log_filename = f"gpu_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        print(f"Saving performance data to {log_filename}...")
        
        with open(log_filename, "w") as f:
            f.write("OptiX Multi-GPU Renderer Performance Log\n")
            f.write("========================================\n\n")
            
            for record in monitoring_data:
                f.write(f"Time: {record['timestamp']} | FPS: {record['fps']:.1f}\n")
                for gpu in record['gpu_stats']:
                    f.write(f"  GPU {gpu['id']} ({gpu['name']}): "
                           f"Util: {gpu['gpu_util']:.1f}%, "
                           f"Memory: {gpu['memory_used']:.0f}MB/{gpu['memory_total']:.0f}MB ({gpu['memory_util']:.1f}%), "
                           f"Temp: {gpu['temperature']:.0f}°C\n")
                f.write("\n")
                
        print(f"Performance data saved to {log_filename}")

if __name__ == "__main__":
    main()