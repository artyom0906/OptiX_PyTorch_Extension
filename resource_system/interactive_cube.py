#!/usr/bin/env python3
"""
OptiX PyTorch Extension - Interactive Cube Viewer
A simple tool to help find the cube by adjusting camera parameters
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from time import time
import math

# Ensure our module is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import optix_resource_system as ors

def create_cube():
    """Create a cube mesh with vertices and indices."""
    # Make cube vertices
    vertices = torch.tensor([
        # Front face
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        # Back face
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]
    ], dtype=torch.float32)
    
    # Cube indices (triangles)
    indices = torch.tensor([
        [0, 1, 2], [2, 3, 0],  # Front face
        [1, 5, 6], [6, 2, 1],  # Right face
        [5, 4, 7], [7, 6, 5],  # Back face
        [4, 0, 3], [3, 7, 4],  # Left face
        [3, 2, 6], [6, 7, 3],  # Top face
        [4, 5, 1], [1, 0, 4]   # Bottom face
    ], dtype=torch.int32)
    
    # Generate UV coordinates (simple planar mapping)
    uv = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
    ], dtype=torch.float32)
    
    # Generate normals (outward from center)
    normals = torch.tensor([
        [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    return vertices, indices, uv, normals

def main():
    """Interactive mode to help find the cube."""
    print("\n=== OptiX Cube Interactive Mode ===\n")
    
    try:
        # Create the resource manager
        print("Creating resource manager...")
        resource_manager = ors.ResourceManager()
        
        # Create renderer
        print("Creating renderer...")
        renderer = ors.Renderer(resource_manager)
        
        # Initialize renderer
        print("Initializing renderer...")
        if not renderer.initialize():
            print("Failed to initialize renderer")
            return
        
        # Create cube geometry
        print("Creating cube geometry...")
        vertices, indices, uv, normals = create_cube()
        cube_geometry = resource_manager.create_geometry(vertices, indices, normals, uv)
        print(f"Cube geometry created with handle: {cube_geometry}")
        
        # Create material
        print("Creating material...")
        material = resource_manager.create_material(ors.MaterialType.LAMBERTIAN)
        
        # Create geometry instance - try with scale to make it easier to see
        print("Creating geometry instance...")
        cube = ors.GeometryInstance(resource_manager, cube_geometry, material)
        cube.set_transform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        
        # Add instance to the renderer
        print("Adding instance to scene...")
        renderer.add_instance(cube)
        
        # Set renderer settings for quick rendering
        settings = ors.RendererSettings()
        settings.max_bounces = 1
        settings.samples_per_pixel = 1
        renderer.set_settings(settings)
        
        # Image dimensions - smaller for faster rendering
        width, height = 400, 300
        
        # Interactive loop
        distance = 4.0
        cam_x, cam_y, cam_z = 0.0, 0.0, -distance
        fov = 60.0
        
        print("\nInteractive Camera Mode")
        print("=======================")
        print("Use the following commands to move the camera:")
        print("  w/s - Move camera forward/backward")
        print("  a/d - Move camera left/right")
        print("  q/e - Move camera up/down")
        print("  +/- - Increase/decrease field of view")
        print("  r   - Reset camera")
        print("  p   - Print current camera settings")
        print("  x   - Exit program")
        
        while True:
            # Set up camera
            camera = ors.CameraParameters()
            camera.position = [cam_x, cam_y, cam_z]
            camera.look_at = [0.0, 0.0, 0.0]  # Always look at origin
            camera.up = [0.0, 1.0, 0.0]
            camera.fov = fov
            
            renderer.set_camera(camera)
            
            print(f"\nCamera pos: ({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f}), FOV: {fov:.1f}")
            
            # Render
            start_time = time()
            image = renderer.render(width, height)
            render_time = time() - start_time
            print(f"Rendering took {render_time:.4f} seconds")
            
            # Convert to numpy and display
            if torch.cuda.is_available():
                image_np = image.cpu().numpy()
            else:
                image_np = image.numpy()
            
            # Display without blocking
            plt.figure(figsize=(8, 6))
            plt.imshow(image_np)
            plt.title(f"Camera: ({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f}), FOV: {fov:.1f}")
            plt.grid(False)
            plt.axis('off')
            plt.draw()
            plt.pause(0.001)
            
            # Get command
            cmd = input("\nEnter command (w/s/a/d/q/e/+/-/r/p/x): ").lower()
            
            step = 0.5  # Movement step size
            
            if cmd == 'x':
                print("Exiting...")
                break
            elif cmd == 'w':
                cam_z += step
            elif cmd == 's':
                cam_z -= step
            elif cmd == 'a':
                cam_x -= step
            elif cmd == 'd':
                cam_x += step
            elif cmd == 'q':
                cam_y += step
            elif cmd == 'e':
                cam_y -= step
            elif cmd == '+':
                fov = min(fov + 5, 120)
            elif cmd == '-':
                fov = max(fov - 5, 20)
            elif cmd == 'r':
                cam_x, cam_y, cam_z = 0.0, 0.0, -distance
                fov = 60.0
            elif cmd == 'p':
                print(f"Camera position: ({cam_x}, {cam_y}, {cam_z})")
                print(f"Camera FOV: {fov}")
            else:
                print("Unknown command. Try again.")
            
            # Close all figures except the current one
            plt.close()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()