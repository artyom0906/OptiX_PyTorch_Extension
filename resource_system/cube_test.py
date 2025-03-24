#!/usr/bin/env python3
"""
OptiX PyTorch Extension - Basic Cube Test

This script provides a very simple test to render a cube with the OptiX renderer.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Import our extension module
import optix_resource_system as ors


def main():
    """Main test function."""
    print("\n=== OptiX Basic Cube Test ===\n")
    
    try:
        # Create the resource manager
        print("Creating resource manager...")
        resource_manager = ors.ResourceManager()
        
        # Create a simple cube geometry
        print("Creating cube geometry...")
        vertices = torch.tensor([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
        ], dtype=torch.float32)
        
        indices = torch.tensor([
            [0, 1, 2], [2, 3, 0],  # Front face
            [1, 5, 6], [6, 2, 1],  # Right face
            [5, 4, 7], [7, 6, 5],  # Back face
            [4, 0, 3], [3, 7, 4],  # Left face
            [3, 2, 6], [6, 7, 3],  # Top face
            [4, 5, 1], [1, 0, 4]   # Bottom face
        ], dtype=torch.int32)
        
        # Create the geometry resource
        geometry = resource_manager.create_geometry(vertices, indices)
        print(f"Geometry created with handle: {geometry}")
        
        # Create a material
        print("Creating material...")
        material = resource_manager.create_material(ors.LegacyMaterialType.LAMBERTIAN)
        print(f"Material created with handle: {material}")
        
        # Create a renderer
        print("Creating renderer...")
        renderer = ors.Renderer(resource_manager)
        print("Initializing renderer...")
        if not renderer.initialize():
            raise RuntimeError("Failed to initialize renderer")
        
        # Configure renderer settings
        width, height = 512, 512
        settings = ors.RendererSettings()
        settings.samples_per_pixel = 1
        settings.max_bounces = 1
        renderer.set_settings(settings)
        
        # Create a geometry instance
        print("Creating geometry instance...")
        instance = ors.GeometryInstance(resource_manager, geometry, material)
        
        # Configure camera
        print("Setting up camera...")
        camera = ors.CameraParameters()
        
        # Just set as individual values to avoid format issues
        camera.position = (0.0, 0.0, 3.0)  # 3 units back along Z
        camera.look_at = (0.0, 0.0, 0.0)   # looking at origin
        camera.up = (0.0, 1.0, 0.0)        # Y is up
        camera.fov = 60.0
        camera.aspect_ratio = float(width) / float(height)
        
        renderer.set_camera(camera)
        
        # Add the instance to the renderer
        print("Adding instance to renderer...")
        renderer.add_instance(instance)
        
        # Render the scene
        print("Rendering scene...")
        start_time = time()
        image = renderer.render(width, height)
        end_time = time()
        
        print(f"Rendering completed in {end_time - start_time:.2f} seconds")
        
        # Save the image
        np_image = image.detach().cpu().numpy()
        np_image = np.clip(np_image, 0, 1)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(np_image)
        plt.axis('off')
        plt.title("Rendered Cube")
        plt.savefig("cube_test.png", bbox_inches='tight')
        plt.show()
        
        print("Image saved as cube_test.png")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()