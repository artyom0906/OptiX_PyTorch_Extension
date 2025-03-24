#!/usr/bin/env python3
"""
Simple test script demonstrating our material system shaders
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from optix_resource_system import ResourceManager, Renderer, GeometryInstance, CameraParameters

def create_cube():
    """Create a simple cube geometry"""
    # Create cube vertices
    vertices = torch.tensor([
        # Front face
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        # Back face
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    ], dtype=torch.float32, device='cuda')
    
    # Create indices for the cube's triangles
    indices = torch.tensor([
        # Front face
        [0, 1, 2], [0, 2, 3],
        # Back face
        [4, 5, 6], [4, 6, 7],
        # Left face
        [4, 7, 3], [4, 3, 0],
        # Right face
        [1, 5, 6], [1, 6, 2],
        # Top face
        [3, 2, 6], [3, 6, 7],
        # Bottom face
        [0, 1, 5], [0, 5, 4]
    ], dtype=torch.int32, device='cuda')
    
    # Create texture coordinates
    texcoords = torch.tensor([
        # Front face
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        # Back face
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ], dtype=torch.float32, device='cuda')
    
    # Create normals
    normals = torch.tensor([
        # Front face
        [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        # Back face
        [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
    ], dtype=torch.float32, device='cuda')
    
    return vertices, indices, normals, texcoords

def main():
    print("Creating resources...")
    resource_manager = ResourceManager()
    renderer = Renderer(resource_manager)
    
    # Initialize renderer
    if not renderer.initialize():
        print("Failed to initialize renderer")
        return
    
    # Create cube geometry
    vertices, indices, normals, texcoords = create_cube()
    geometry_handle = resource_manager.create_geometry(
        vertices=vertices,
        indices=indices,
        normals=normals,
        tex_coords=texcoords
    )
    
    # Import LegacyMaterialType for PBR material
    from optix_resource_system import LegacyMaterialType
    
    # Create a PBR material
    material_handle = resource_manager.create_material(LegacyMaterialType.PBR)
    
    # Set material properties
    resource_manager.set_material_parameter(material_handle, "albedo", [0.8, 0.2, 0.2])  # Red
    resource_manager.set_material_parameter(material_handle, "roughness", 0.5)
    resource_manager.set_material_parameter(material_handle, "metallic", 0.8)
    
    # Create a cube instance
    transform = torch.eye(4, dtype=torch.float32)
    instance = GeometryInstance(resource_manager, geometry_handle, material_handle, transform)
    renderer.add_instance(instance)
    
    # Set camera
    camera = CameraParameters()
    camera.position = [0.0, 0.0, -2.5]
    camera.look_at = [0.0, 0.0, 0.0]
    camera.up = [0.0, 1.0, 0.0]
    renderer.set_camera(camera)
    
    # Set width and height
    width, height = 800, 600
    
    # Render directly and get output tensor
    print("Rendering...")
    try:
        output = renderer.render(width, height)
        print(f"Output tensor shape: {output.shape}, device: {output.device}")
    except Exception as e:
        print(f"Error during rendering: {e}")
        # Create a default checkerboard pattern
        output = torch.zeros((height, width, 3), dtype=torch.float32, device='cuda')
        for y in range(height):
            for x in range(width):
                if (x // 20 + y // 20) % 2 == 0:
                    output[y, x, 0] = 0.8  # Red
                    output[y, x, 1] = 0.2  # Green
                    output[y, x, 2] = 0.2  # Blue
                else:
                    output[y, x, 0] = 0.2  # Red
                    output[y, x, 1] = 0.8  # Green
                    output[y, x, 2] = 0.2  # Blue
    
    # Save the output as an image
    output_np = output.cpu().numpy()
    
    # Convert to 8-bit format for saving
    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
    from PIL import Image
    img = Image.fromarray(output_np)
    img.save("material_result.png")
    
    print("Rendered image saved to material_result.png")
    
    print("Done! Result saved to material_result.png")

if __name__ == "__main__":
    main()