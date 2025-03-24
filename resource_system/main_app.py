#!/usr/bin/env python3
"""
OptiX Resource System - Simple Cube Application

This script creates a simple cube and allows camera movement around it.
Simplified version with just the core functionality.
Uses minimal shader that colors collisions red.
Includes GPU monitoring for resource usage tracking.
"""

import math
import time
import numpy as np
import torch
import pygame
import matplotlib.pyplot as plt
import sys
import os
import GPUtil
import psutil
from datetime import datetime

# Add parent directory to path to import Python modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import modules from the parent directory
from python.controllers.KeyboardController import KeyboardController  # Use keyboard controls
from python.models.Camera import Camera
from python.models.Player import Player

# Import resource system
import optix_resource_system as ors

# Initialize pygame
pygame.init()

# Create a window - smaller for better performance
width, height = 800, 600  
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("OptiX Cube Viewer")

def create_floor():
    """Create a floor plane geometry"""
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
    
    print("Created floor plane with upward-facing normal")
    return vertices, indices

def create_cube():
    """Create a simple cube geometry"""
    # Cube vertices
    vertices = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # front face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # back face
    ], dtype=torch.float32)
    
    # Cube indices (triangles) - reversed winding order to face outward
    indices = torch.tensor([
        # Change winding order to make triangles face outward
        [0, 2, 1], [0, 3, 2],  # front face reversed
        [1, 6, 5], [1, 2, 6],  # right face reversed
        [5, 7, 4], [5, 6, 7],  # back face reversed
        [4, 3, 0], [4, 7, 3],  # left face reversed
        [3, 6, 2], [3, 7, 6],  # top face reversed
        [4, 1, 5], [4, 0, 1]   # bottom face reversed
    ], dtype=torch.int32)
    
    print("Created cube with outward-facing vertices")
    return vertices, indices

def main():
    """Main function"""
    print("Starting OptiX Cube Viewer...")
    
    # Create resource manager
    print("Creating resource manager...")
    resource_manager = ors.ResourceManager()
    
    # Create renderer
    print("Initializing renderer...")
    renderer = ors.Renderer(resource_manager)
    if not renderer.initialize():
        print("Failed to initialize renderer")
        return
    
    # Create cube geometry
    print("Creating cube geometry...")
    vertices, indices = create_cube()
    cube_geometry = resource_manager.create_geometry(vertices, indices)
    
    # Create a simple diffuse material for cube
    print("Creating cube material...")
    # Use LAMBERTIAN material type with blue color
    cube_material = resource_manager.create_material(ors.MaterialType.LAMBERTIAN)
    # Set blue color parameter using our new Python binding
    resource_manager.set_material_parameter(cube_material, "albedo", [0.2, 0.4, 0.8])
    
    # Create single cube instance
    print("Creating cube instance...")
    cube_instance = ors.GeometryInstance(
        resource_manager,
        cube_geometry,
        cube_material
    )
    
    # Add the cube instance to the renderer
    renderer.add_instance(cube_instance)
    
    # Place the cube in the center of the scene
    # Position it a bit higher to ensure it's above the floor
    cube_instance.set_transform(
        [0.0, 0.5, -6.0],                              # position - further back at z=-6
        [np.radians(0), np.radians(0), np.radians(0)], # no rotation
        [1.0, 1.0, 1.0]                                # larger scale to be more visible
    )
    
    # Create floor geometry
    print("Creating floor geometry...")
    floor_vertices, floor_indices = create_floor()
    floor_geometry = resource_manager.create_geometry(floor_vertices, floor_indices)
    
    # Create a different colored material for floor
    print("Creating floor material...")
    floor_material = resource_manager.create_material(ors.MaterialType.LAMBERTIAN)
    # Set green color parameter using our new Python binding
    resource_manager.set_material_parameter(floor_material, "albedo", [0.2, 0.8, 0.2])
    
    # Create floor instance
    print("Creating floor instance...")
    floor_instance = ors.GeometryInstance(
        resource_manager,
        floor_geometry,
        floor_material
    )
    
    # Add the floor instance to the renderer
    renderer.add_instance(floor_instance)
    
    # Place the floor under the cube, centered at the same position
    # Make sure it's well below the cube which is at y=0.5
    floor_instance.set_transform(
        [0.0, -1.0, -6.0],                            # position below cube at same z-position as cube
        [np.radians(0), np.radians(0), np.radians(0)], # no rotation
        [6.0, 0.1, 6.0]                               # Make it very wide but thin
    )
    
    # Set up camera and controls using both keyboard and gamepad
    # Try to use gamepad first, fall back to keyboard if no gamepad is available
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        from python.controllers.GamepadController import GamepadController
        controller = GamepadController()
        print("Gamepad detected - using gamepad controls")
    else:
        controller = KeyboardController()
        print("No gamepad detected - using keyboard controls")
    # Create camera and ensure all vectors are on the same device
    device = torch.device("cuda:0")  # Explicitly use CUDA device 0
    camera = Camera(
        torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32),  # Position camera higher at the origin
        torch.tensor([0.0, 0.5, -6.0], device=device, dtype=torch.float32), # Look at the cube position
        torch.tensor([0, 1, 0], device=device, dtype=torch.float32)         # Up vector
    )
    player = Player(controller, camera)
    
    # Camera parameters for renderer - point directly at cube
    camera_params = ors.CameraParameters()
    camera_params.position = camera.position.tolist()
    camera_params.look_at = camera.lookat.tolist()  # Use camera's actual lookat target
    camera_params.up = camera.up.tolist()
    camera_params.fov = 120.0  # Extra wide field of view to see more
    camera_params.aspect_ratio = width / height
    
    # Debug the camera setup
    print(f"Camera setup:")
    print(f"  Position: {camera_params.position}")
    print(f"  Look At: {camera_params.look_at}")
    print(f"  Up: {camera_params.up}")
    print(f"  FOV: {camera_params.fov} degrees")
    print(f"  Aspect Ratio: {camera_params.aspect_ratio}")
    
    renderer.set_camera(camera_params)
    
    # Print controls explanation
    if pygame.joystick.get_count() > 0:
        print("\nGAMEPAD CONTROLS:")
        print("  Left stick       - Move forward/backward/left/right")
        print("  Right stick      - Look around")
        print("  Left/Right Trig. - Move up/down")
    else:
        print("\nKEYBOARD CONTROLS:")
        print("  W/S        - Move forward/backward")
        print("  A/D        - Move left/right")
        print("  Q/E        - Move up/down")
        print("  Arrow keys - Look around")
    print("  ESC        - Exit program\n")
    
    # Set up GPU monitoring
    gpu_stats = []
    def get_gpu_stats():
        """Get current GPU stats"""
        gpus = GPUtil.getGPUs()
        stats = []
        for i, gpu in enumerate(gpus):
            stats.append({
                'id': i,
                'name': gpu.name,
                'load': gpu.load * 100,  # Convert to percentage
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_util': gpu.memoryUtil * 100,  # Convert to percentage
                'temperature': gpu.temperature
            })
        return stats
    
    def log_resources():
        """Log system resources to console"""
        # Get current timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        ram_percent = psutil.virtual_memory().percent
        
        # Get GPU stats
        gpu_data = get_gpu_stats()
        
        # Log data
        print(f"[{timestamp}] SYSTEM RESOURCES:")
        print(f"  CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}%")
        for gpu in gpu_data:
            print(f"  GPU{gpu['id']} ({gpu['name']}): {gpu['load']:.1f}% load | "
                  f"Memory: {gpu['memory_used']:.0f}MB/{gpu['memory_total']:.0f}MB ({gpu['memory_util']:.1f}%) | "
                  f"Temp: {gpu['temperature']}°C")
        
        # Save data for plotting/analysis
        gpu_stats.append({
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'ram_percent': ram_percent,
            'gpu_data': gpu_data
        })
    
    # Main render loop
    running = True
    clock = pygame.time.Clock()
    frame_times = [0.0]
    log_interval = 2.0  # Log resources every 2 seconds
    last_log_time = time.time()
    
    print("Entering main loop...")
    while running:
        # Record the start time for FPS calculation
        start_time = time.perf_counter()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update player position/camera
        player.update(frame_times[-1])
        
        # Fully recreate camera parameters each frame to ensure basis vectors are recalculated
        camera_params = ors.CameraParameters()
        camera_params.position = camera.position.tolist()
        camera_params.look_at = camera.lookat.tolist()  # Use camera's actual lookat target
        camera_params.up = camera.up.tolist()
        camera_params.fov = 120.0  # Extra wide field of view to see more
        camera_params.aspect_ratio = width / height
        renderer.set_camera(camera_params)
        
        # No need to force scene update - the camera parameters will trigger necessary updates
        
        # Log resources periodically
        current_time = time.time()
        if current_time - last_log_time >= log_interval:
            log_resources()
            last_log_time = current_time
        
        # Debug camera position during movement (less frequently now)
        if frame_times[-1] % 2.0 < 0.05:  # Log every 2 seconds
            print(f"Camera position: {camera_params.position}")
        
        # Render the scene
        output_image = renderer.render(width, height)
        
        # Convert tensor to numpy array
        np_image = output_image.cpu().numpy()
        
        # Save a matplotlib version for comparison
        plt.figure(figsize=(8, 8))
        plt.imshow(np_image)
        plt.axis('off')
        plt.title("Rendered Cube")
        plt.savefig("cube_test.png", bbox_inches='tight')
        # plt.show()  # Comment this out to avoid blocking the game loop
        
        # For PyGame, ensure proper scaling and RGB format
        pygame_image = np.transpose(np_image, (1, 0, 2))  # PyGame expects height, width, channels
        # Scale to 0-255 uint8 for PyGame
        pygame_image = (pygame_image * 255).astype(np.uint8)
        # Create PyGame surface
        surface = pygame.surfarray.make_surface(pygame_image[:, :, :3])
        
        # Display the rendered image
        window.blit(surface, (0, 0))
        
        # Display instructions
        font = pygame.font.Font(None, 24)
        if pygame.joystick.get_count() > 0:
            text = font.render("Controls: Left stick=move, Right stick=look, LT/RT=up/down", True, (255, 255, 255))
        else:
            text = font.render("Controls: WASD=move, QE=up/down, Arrows=look", True, (255, 255, 255))
        window.blit(text, (10, 10))
        
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
            pygame.display.set_caption(f"OptiX Cube Viewer - FPS: {fps:.2f}")
        
        # Limit to 60 FPS
        clock.tick(60)
        #return
    
    # Clean up
    pygame.quit()
    print("Application closed")
    
    # Save GPU statistics to file for later analysis
    if len(gpu_stats) > 0:
        print(f"Saving {len(gpu_stats)} GPU monitoring records to gpu_stats.txt")
        with open("gpu_stats.txt", "w") as f:
            for stat in gpu_stats:
                f.write(f"Time: {stat['timestamp']}, CPU: {stat['cpu_percent']:.1f}%, RAM: {stat['ram_percent']:.1f}%\n")
                for gpu in stat['gpu_data']:
                    f.write(f"  GPU{gpu['id']}: {gpu['load']:.1f}% load, {gpu['memory_used']:.0f}MB used, {gpu['memory_util']:.1f}% util\n")
        print("Resource statistics saved successfully")

if __name__ == "__main__":
    main()