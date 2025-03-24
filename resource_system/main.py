#!/usr/bin/env python3
"""
OptiX PyTorch Extension - Resource System Test Script

This script demonstrates the usage of the OptiX Resource System through its Python API.
It creates resources, sets up a scene, and renders an image using the path tracer.
Includes GPU monitoring for performance tracking.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
import GPUtil
import psutil
from datetime import datetime

# Import our extension
import optix_resource_system as ors


def create_cube():
    """Create a cube mesh with vertices and indices."""
    # Cube vertices
    vertices = torch.tensor([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
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
    
    return vertices, indices


def create_checker_texture(size=256, check_size=32):
    """Create a checker pattern texture."""
    # Create a blank texture
    texture = torch.zeros((size, size, 4), dtype=torch.float32)
    
    # Fill with a checker pattern
    for i in range(size):
        for j in range(size):
            value = 1.0 if ((i // check_size + j // check_size) % 2 == 0) else 0.1
            texture[i, j, 0] = value  # R
            texture[i, j, 1] = value  # G
            texture[i, j, 2] = value  # B
            texture[i, j, 3] = value  # B
    
    return texture


def setup_camera(width, height):
    """Create and configure a camera with a better position for viewing the cube."""
    camera = ors.CameraParameters()
    # Position the camera clearly in front of the cube for better visibility
    camera.position = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float32)
    camera.look_at = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    camera.up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    # Use a wider field of view to ensure the cube is in view
    camera.fov = 70.0
    camera.aspect_ratio = width / height
    # Print camera info for debugging
    print(f"Camera configured:")
    print(f"  Position: ({camera.position[0]}, {camera.position[1]}, {camera.position[2]})")
    print(f"  Look At: ({camera.look_at[0]}, {camera.look_at[1]}, {camera.look_at[2]})")
    print(f"  FOV: {camera.fov} degrees")
    return camera


def save_image(tensor, filename="output.png"):
    """Save a tensor as an image."""
    # Convert to numpy and ensure values are in [0, 1]
    np_image = tensor.detach().cpu().numpy()
    np_image = np.clip(np_image, 0, 1)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Image saved to {filename}")


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

def log_resources(label=""):
    """Log system resources to console with optional label"""
    # Get current timestamp
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_percent = psutil.virtual_memory().percent
    
    # Get GPU stats
    gpu_data = get_gpu_stats()
    
    # Log data
    label_str = f" - {label}" if label else ""
    print(f"[{timestamp}] SYSTEM RESOURCES{label_str}:")
    print(f"  CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}%")
    for gpu in gpu_data:
        print(f"  GPU{gpu['id']} ({gpu['name']}): {gpu['load']:.1f}% load | "
              f"Memory: {gpu['memory_used']:.0f}MB/{gpu['memory_total']:.0f}MB ({gpu['memory_util']:.1f}%) | "
              f"Temp: {gpu['temperature']}°C")
    
    return {
        'timestamp': timestamp,
        'cpu_percent': cpu_percent,
        'ram_percent': ram_percent,
        'gpu_data': gpu_data
    }

def main():
    """Main function demonstrating the resource system."""
    print("\n=== OptiX Resource System Test with GPU Monitoring ===\n")
    
    # Store resource stats
    resource_stats = []
    
    try:
        # Log initial system state
        print("Initial system state:")
        resource_stats.append(log_resources("startup"))
        
        # Create the resource manager - central point for all resources
        print("Creating resource manager...")
        resource_manager = ors.ResourceManager()
        
        # Print available devices
        devices = resource_manager.get_available_devices()
        print(f"Available devices: {devices}")
        
        # Create a simple cube geometry
        print("Creating cube geometry...")
        vertices, indices = create_cube()
        cube_geometry = resource_manager.create_geometry(vertices, indices)
        print(f"Cube geometry created with handle: {cube_geometry}")
        resource_stats.append(log_resources("after geometry creation"))
        
        # Create a checker texture
        print("Creating checker texture...")
        texture = create_checker_texture()
        checker_texture = resource_manager.create_texture(texture)
        print(f"Checker texture created with handle: {checker_texture}")
        resource_stats.append(log_resources("after texture creation"))
        
        # Create materials
        print("Creating materials...")
        lambertian_material = resource_manager.create_material(ors.MaterialType.LAMBERTIAN)
        print(f"Lambertian material created with handle: {lambertian_material}")
        
        mirror_material = resource_manager.create_material(ors.MaterialType.MIRROR)
        print(f"Mirror material created with handle: {mirror_material}")
        resource_stats.append(log_resources("after material creation"))
        
        # Create a renderer
        print("Creating renderer...")
        renderer = ors.Renderer(resource_manager)
        
        # Initialize the renderer
        print("Initializing renderer...")
        if renderer.initialize():
            print("Renderer initialized successfully")
        else:
            print("Failed to initialize renderer")
            return
        
        resource_stats.append(log_resources("after renderer init"))
        
        # Set rendering width and height
        width, height = 640, 480
        
        # Create instances for multiple objects
        print("Creating geometry instances...")
        # Main cube at the center
        cube_instance = ors.GeometryInstance(
            resource_manager,
            cube_geometry,
            lambertian_material
        )
        
        # Make the cube easier to see - rotate it 45 degrees around Y axis
        transform = torch.eye(4, dtype=torch.float32)
        # Apply 45-degree rotation around Y axis
        angle = torch.tensor(45.0 * torch.pi / 180.0)
        transform[0, 0] = torch.cos(angle)
        transform[0, 2] = torch.sin(angle)
        transform[2, 0] = -torch.sin(angle)
        transform[2, 2] = torch.cos(angle)
        
        # Print transform matrix for debugging
        print("Cube transform matrix:")
        for row in range(4):
            print(f"  [{transform[row, 0]:.2f}, {transform[row, 1]:.2f}, {transform[row, 2]:.2f}, {transform[row, 3]:.2f}]")
        
        cube_instance.set_transform(transform)
        
        # Add the cube instance to the renderer
        renderer.add_instance(cube_instance)
        print("Added main cube instance")
        
        # Add mirrored cube to the left
        mirror_instance = ors.GeometryInstance(
            resource_manager,
            cube_geometry,
            mirror_material
        )
        # Position to the left
        mirror_transform = torch.eye(4, dtype=torch.float32)
        mirror_transform[0, 3] = -2.5  # X translation
        mirror_instance.set_transform(mirror_transform)
        renderer.add_instance(mirror_instance)
        print("Added mirror cube instance")
        
        resource_stats.append(log_resources("after scene setup"))
        
        # Set camera
        camera = setup_camera(width, height)
        renderer.set_camera(camera)
        print("Camera configured")
        
        # Configure renderer settings
        settings = ors.RendererSettings()
        settings.samples_per_pixel = 4
        settings.max_bounces = 4
        settings.denoise_result = True  # Enable denoising if available
        renderer.set_settings(settings)
        print("Renderer settings configured")
        
        # Render the scene
        print("\nRendering scene...")
        start_time = time()
        resource_stats.append(log_resources("before rendering"))
        
        image = renderer.render(width, height)
        
        end_time = time()
        resource_stats.append(log_resources("after rendering"))
        print(f"Rendering completed in {end_time - start_time:.2f} seconds")
        print(f"Image shape: {image.shape}")
        
        # Save the rendered image
        save_image(image, "optix_render.png")
        
        # Save resource statistics
        print(f"\nSaving {len(resource_stats)} resource monitoring records to resource_stats.txt")
        with open("resource_stats.txt", "w") as f:
            for i, stat in enumerate(resource_stats):
                f.write(f"--- Record {i+1} - Time: {stat['timestamp']} ---\n")
                f.write(f"CPU: {stat['cpu_percent']:.1f}%, RAM: {stat['ram_percent']:.1f}%\n")
                for gpu in stat['gpu_data']:
                    f.write(f"GPU{gpu['id']} ({gpu['name']}): {gpu['load']:.1f}% load, "
                           f"{gpu['memory_used']:.0f}MB/{gpu['memory_total']:.0f}MB ({gpu['memory_util']:.1f}%), "
                           f"Temp: {gpu['temperature']}°C\n")
                f.write("\n")
        
        print("\nResource system test completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

