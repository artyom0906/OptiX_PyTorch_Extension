#!/usr/bin/env python3
"""
Material system example for OptiX PyTorch Extension.

This script demonstrates the use of the enhanced material system
with PBR material support and MDL compatibility.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from optix_resource_system import (
    ResourceManager, Renderer, CameraParameters, 
    RendererSettings, GeometryInstance, MaterialSystem,
    LegacyMaterialType, MaterialFlags, TextureType
)

def create_textured_cube(resource_manager, material_system):
    """Create a cube with a PBR material and textures."""
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
    
    # Create geometry
    geometry_handle = resource_manager.create_geometry(
        vertices=vertices,
        indices=indices,
        normals=normals,
        tex_coords=texcoords
    )
    
    # Generate a simple procedural texture
    size = 256
    albedo_texture = torch.zeros((size, size, 4), dtype=torch.float32, device='cuda')
    metallic_roughness = torch.zeros((size, size, 4), dtype=torch.float32, device='cuda')
    
    # Create a checkerboard pattern
    for i in range(size):
        for j in range(size):
            if (i // 32 + j // 32) % 2 == 0:
                albedo_texture[i, j, 0] = 0.8  # R
                albedo_texture[i, j, 1] = 0.2  # G
                albedo_texture[i, j, 2] = 0.2  # B
                
                metallic_roughness[i, j, 0] = 0.9  # Metallic
                metallic_roughness[i, j, 1] = 0.2  # Roughness
            else:
                albedo_texture[i, j, 0] = 0.2  # R
                albedo_texture[i, j, 1] = 0.8  # G
                albedo_texture[i, j, 2] = 0.2  # B
                
                metallic_roughness[i, j, 0] = 0.1  # Metallic
                metallic_roughness[i, j, 1] = 0.8  # Roughness
            
            # Set alpha to 1.0
            albedo_texture[i, j, 3] = 1.0
            metallic_roughness[i, j, 3] = 1.0
    
    # Create textures
    albedo_tex_handle = resource_manager.create_texture(albedo_texture, TextureType.RGBA)
    mr_tex_handle = resource_manager.create_texture(metallic_roughness, TextureType.RGBA)
    
    # Create PBR material
    material_handle = material_system.create_material_instance(2)  # PBR material
    
    # Set material parameters
    material_system.set_parameter(material_handle, "albedo", [0.9, 0.9, 0.9])
    material_system.set_parameter(material_handle, "roughness", 0.5)
    material_system.set_parameter(material_handle, "metallic", 0.0)
    material_system.set_parameter(material_handle, "specular", 0.5)
    
    # Set textures
    material_system.set_texture_parameter(material_handle, "albedo", albedo_tex_handle)
    material_system.set_texture_parameter(material_handle, "metallicRoughness", mr_tex_handle)
    
    # Create a GeometryInstance using the geometry and material
    transform = torch.eye(4, dtype=torch.float32)
    instance = GeometryInstance(resource_manager, geometry_handle, material_handle, transform)
    
    return instance

def create_scene(resource_manager, renderer):
    """Create the scene with multiple materials."""
    # Create material system
    material_system = MaterialSystem()
    material_system.initialize()
    
    # Create a PBR cube
    cube = create_textured_cube(resource_manager, material_system)
    renderer.add_instance(cube)
    
    # Set up camera
    camera = CameraParameters()
    camera.position = [0.0, 0.0, -3.0]
    camera.look_at = [0.0, 0.0, 0.0]
    camera.up = [0.0, 1.0, 0.0]
    camera.fov = 45.0
    renderer.set_camera(camera)
    
    # Set up renderer settings
    settings = RendererSettings()
    settings.samples_per_pixel = 1
    settings.max_bounces = 3
    renderer.set_settings(settings)

def main():
    # Create resource manager and renderer
    resource_manager = ResourceManager()
    renderer = Renderer(resource_manager, 0)  # Use GPU 0
    
    # Initialize the renderer
    if not renderer.initialize():
        print("Failed to initialize renderer")
        return
    
    # Create the scene
    create_scene(resource_manager, renderer)
    
    # Set up output tensor
    width, height = 800, 600
    output = torch.zeros((height, width, 3), dtype=torch.float32, device='cuda')
    
    # Render
    print("Rendering...")
    start_time = time.time()
    renderer.render(output)
    end_time = time.time()
    print(f"Rendering took {end_time - start_time:.2f} seconds")
    
    # Save the image
    output_np = output.cpu().numpy()
    output_np = np.clip(output_np, 0.0, 1.0)  # Clamp to [0, 1]
    plt.figure(figsize=(10, 7.5))
    plt.imshow(output_np)
    plt.axis('off')
    plt.savefig("material_example.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    main()