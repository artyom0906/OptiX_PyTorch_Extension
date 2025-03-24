"""
OBJ Loader for OptiX Resource System

This module provides a class for loading OBJ files into the OptiX Resource System.
It includes support for loading textures, normal maps, and other material properties.
"""

import os
import torch
import numpy as np
from PIL import Image
import trimesh
from typing import Optional, Tuple, Dict, Any, List, Union

import optix_resource_system as ors
from resource_system.geometry import Geometry

def compute_tangents(vertices: torch.Tensor, 
                    uv: torch.Tensor, 
                    indices: torch.Tensor, 
                    normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-vertex tangents and bitangents.

    Args:
        vertices (torch.Tensor): Vertex positions, shape (V, 3)
        uv (torch.Tensor): Texture coordinates, shape (V, 2)
        indices (torch.Tensor): Triangle indices, shape (F, 3)
        normals (torch.Tensor): Vertex normals, shape (V, 3)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Per-vertex tangents and bitangents, both shape (V, 3)
    """
    dtype = vertices.dtype

    # Extract vertex positions and UVs for each triangle
    v0 = vertices[indices[:, 0]]  # (F, 3)
    v1 = vertices[indices[:, 1]]  # (F, 3)
    v2 = vertices[indices[:, 2]]  # (F, 3)

    uv0 = uv[indices[:, 0]]       # (F, 2)
    uv1 = uv[indices[:, 1]]       # (F, 2)
    uv2 = uv[indices[:, 2]]       # (F, 2)

    # Compute edge vectors
    edge1 = v1 - v0                 # (F, 3)
    edge2 = v2 - v0                 # (F, 3)

    # Compute delta UVs
    delta_uv1 = uv1 - uv0           # (F, 2)
    delta_uv2 = uv2 - uv0           # (F, 2)

    # Compute the denominator of the tangent/bitangent equation
    denominator = delta_uv1[:, 0] * delta_uv2[:, 1] - delta_uv2[:, 0] * delta_uv1[:, 1]  # (F,)
    # To avoid division by zero, set invalid denominators to a small value
    epsilon = 1e-8
    f = torch.where(denominator != 0, 1.0 / denominator, torch.tensor(0.0))

    # Compute tangents and bitangents
    tangent = (delta_uv2[:, 1].unsqueeze(1) * edge1 - delta_uv1[:, 1].unsqueeze(1) * edge2) * f.unsqueeze(1)  # (F, 3)
    bitangent = (-delta_uv2[:, 0].unsqueeze(1) * edge1 + delta_uv1[:, 0].unsqueeze(1) * edge2) * f.unsqueeze(1)  # (F, 3)

    # Normalize tangents and bitangents
    tangent = torch.nn.functional.normalize(tangent, p=2, dim=1)  # (F, 3)
    bitangent = torch.nn.functional.normalize(bitangent, p=2, dim=1)  # (F, 3)

    # Initialize per-vertex tangent and bitangent accumulators
    V = vertices.shape[0]
    tangents = torch.zeros((V, 3), dtype=dtype)
    bitangents = torch.zeros((V, 3), dtype=dtype)

    # Accumulate tangents and bitangents for each vertex
    # Using scatter_add for efficiency
    indices_flat = indices.view(-1)  # (F*3,)

    # Repeat tangents and bitangents for each vertex of the triangle
    tangents_repeated = tangent.repeat(3, 1)      # (F*3, 3)
    bitangents_repeated = bitangent.repeat(3, 1)  # (F*3, 3)

    # Scatter add tangents and bitangents to the corresponding vertices
    tangents = tangents.index_add(0, indices_flat, tangents_repeated)
    bitangents = bitangents.index_add(0, indices_flat, bitangents_repeated)

    # Normalize the accumulated tangents and bitangents
    tangents = torch.nn.functional.normalize(tangents, p=2, dim=1)
    bitangents = torch.nn.functional.normalize(bitangents, p=2, dim=1)

    # Orthogonalize tangents with normals
    # Tangent orthogonalization: T = normalize(T - N * dot(T, N))
    tangents = torch.nn.functional.normalize(tangents - normals * torch.sum(tangents * normals, dim=1, keepdim=True), p=2, dim=1)
    
    # Recompute bitangents to ensure orthogonality: B = cross(N, T)
    bitangents = torch.cross(normals, tangents, dim=1)
    bitangents = torch.nn.functional.normalize(bitangents, p=2, dim=1)

    return tangents, bitangents

class ObjLoader:
    """
    Loads 3D models from .obj files and textures from image files.
    Creates resources in the OptiX Resource System.
    """
    
    def __init__(self, resource_manager):
        """
        Initialize the OBJ loader.
        
        Args:
            resource_manager: The OptiX resource manager used to create resources
        """
        self.resource_manager = resource_manager
    
    def load_texture(self, texture_path: str, flip_y: bool = False) -> int:
        """
        Load a texture from file and create a texture resource.
        
        Args:
            texture_path: Path to the texture image file
            flip_y: Whether to flip the texture vertically
            
        Returns:
            Handle to the created texture resource
        """
        if not os.path.exists(texture_path):
            print(f"Warning: Texture file not found: {texture_path}")
            return None
            
        try:
            # Load and convert the image to RGBA
            image = Image.open(texture_path).convert('RGBA')
            
            # Flip the image if needed
            if flip_y:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                
            # Convert to numpy array, then to tensor
            image_data = np.array(image).astype(np.float32) / 255.0  # Normalize to 0.0-1.0
            texture_tensor = torch.tensor(image_data, dtype=torch.float32, device='cpu')
            
            # Create texture resource
            texture_handle = self.resource_manager.create_texture(texture_tensor, ors.TextureType.RGBA)
            return texture_handle
            
        except Exception as e:
            print(f"Error loading texture {texture_path}: {str(e)}")
            return None
    
    def load_obj(self, 
                obj_path: str, 
                texture_path: Optional[str] = None,
                normal_map_path: Optional[str] = None, 
                metallic_roughness_path: Optional[str] = None,
                emission_texture_path: Optional[str] = None,
                flip_textures: Tuple[bool, bool, bool, bool] = (False, False, False, False)
                ) -> Dict[str, Any]:
        """
        Load an OBJ file and associated textures.
        
        Args:
            obj_path: Path to the OBJ file
            texture_path: Path to the albedo/diffuse texture
            normal_map_path: Path to the normal map texture
            metallic_roughness_path: Path to the metallic-roughness texture
            emission_texture_path: Path to the emission texture
            flip_textures: Tuple of booleans for flipping (albedo, normal, metallic, emission)
            
        Returns:
            Dictionary containing geometry handle, material handle, and instance
        """
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")
        
        print(f"Loading OBJ file: {obj_path}")
        
        # Load the mesh with trimesh
        mesh = trimesh.load(obj_path, force='mesh')
        
        # Extract vertex data
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device='cpu')
        indices = torch.tensor(mesh.faces, dtype=torch.int32, device='cpu')

        # Extract texture coordinates if available
        tex_coords = None
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            tex_coords = torch.tensor(mesh.visual.uv, dtype=torch.float32, device='cpu')

        print(tex_coords, vertices, indices)

        return vertices, indices, tex_coords

        # # Extract vertex normals if available
        # vertex_normals = None
        # if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        #     vertex_normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)


        # # Compute tangent space if we have UVs and normals
        # tangents = None
        # bitangents = None
        # if tex_coords is not None and vertex_normals is not None:
        #     tangents, bitangents = compute_tangents(vertices, tex_coords, indices, vertex_normals)

        #
        # # Create geometry resource
        # print(vertices, indices, tex_coords)
        # print(vertices.device, indices.device)
        # geometry_handle = self.resource_manager.create_geometry(
        #     vertices=vertices,
        #     indices=indices,
        #     #normals=vertex_normals,
        #     #tex_coords=tex_coords,
        #     #tangents=tangents,
        #     #bitangents=bitangents
        # )
        # print(vertices)
        # print(indices)
        #
        # # Load textures
        # albedo_texture = None
        # normal_texture = None
        # metallic_roughness_texture = None
        # emission_texture = None
        #
        # #if texture_path:
        # #    albedo_texture = self.load_texture(texture_path, flip_textures[0])
        #
        # #if normal_map_path:
        # #    normal_texture = self.load_texture(normal_map_path, flip_textures[1])
        # #
        # #if metallic_roughness_path:
        # #    metallic_roughness_texture = self.load_texture(
        # #        metallic_roughness_path, flip_textures[2])
        # #
        # #if emission_texture_path:
        # #    emission_texture = self.load_texture(
        # #        emission_texture_path, flip_textures[3])
        #
        # # Create a material
        # material_handle = self.resource_manager.create_material(ors.LegacyMaterialType.LAMBERTIAN)
        #
        # # Set material parameters
        # # Default color will be white if no texture is provided
        # self.resource_manager.set_material_parameter(material_handle, "albedo", [0.3, 0.2, 0.9])
        #
        # # Assign textures to material if available
        # #if albedo_texture:
        # #    self.resource_manager.set_material_parameter(material_handle, "albedoTexture", albedo_texture)
        #
        # #if normal_texture:
        # #   self.resource_manager.set_material_parameter(material_handle, "normalTexture", normal_texture)
        # #
        # #if metallic_roughness_texture:
        # #    self.resource_manager.set_material_parameter(material_handle, "metallicRoughnessTexture", metallic_roughness_texture)
        # #
        # #if emission_texture:
        # #    self.resource_manager.set_material_parameter(material_handle, "emissionTexture", emission_texture)
        # #    self.resource_manager.set_material_parameter(material_handle, "emission", [1.0, 1.0, 1.0])
        #
        # # Create an instance
        # instance = ors.GeometryInstance(
        #     self.resource_manager,
        #     geometry_handle,
        #     material_handle
        # )
        #
        # # Return all created resources
        # return {
        #     "geometry_handle": geometry_handle,
        #     "material_handle": material_handle,
        #     "instance": instance,
        #     "has_normal_map": normal_texture is not None,
        #     "has_emission": emission_texture is not None,
        #     "has_metallic_roughness": metallic_roughness_texture is not None
        # }

# Example usage:
# resource_manager = ors.ResourceManager()
# loader = ObjLoader(resource_manager)
# result = loader.load_obj(
#     "models/test/untitled.obj", 
#     "models/test/box.png", 
#     flip_textures=(True, True, True, True)
# )
# instance = result["instance"]
# instance.set_transform([0, 0, -5], [0, 0, 0], [1, 1, 1])