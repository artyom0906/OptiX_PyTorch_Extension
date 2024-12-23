import optix_renderer
import torch
import numpy as np
import trimesh
from PIL import Image
from matplotlib import pyplot as plt

from python.loaders.Model import Model
from python.loaders.ModelLoader import ModelLoader

import optix_renderer
import torch
import numpy as np
import trimesh
from PIL import Image
from matplotlib import pyplot as plt

from python.loaders.Model import Model
from python.loaders.ModelLoader import ModelLoader

def compute_tangents(vertices, uv, indices, normals):
    """
    Compute per-vertex tangents and bitangents using PyTorch.

    Args:
        vertices (torch.Tensor): Vertex positions, shape (V, 3)
        uv (torch.Tensor): Texture coordinates, shape (V, 2)
        indices (torch.Tensor): Triangle indices, shape (F, 3)
        normals (torch.Tensor): Vertex normals, shape (V, 3)

    Returns:
        tangents (torch.Tensor): Per-vertex tangents, shape (V, 3)
        bitangents (torch.Tensor): Per-vertex bitangents, shape (V, 3)
    """
    device = vertices.device
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
    f = torch.where(denominator != 0, 1.0 / denominator, torch.tensor(0.0, device=device))

    # Compute tangents and bitangents
    tangent = (delta_uv2[:, 1].unsqueeze(1) * edge1 - delta_uv1[:, 1].unsqueeze(1) * edge2) * f.unsqueeze(1)  # (F, 3)
    bitangent = (-delta_uv2[:, 0].unsqueeze(1) * edge1 + delta_uv1[:, 0].unsqueeze(1) * edge2) * f.unsqueeze(1)  # (F, 3)

    # Normalize tangents and bitangents
    tangent = torch.nn.functional.normalize(tangent, p=2, dim=1)  # (F, 3)
    bitangent = torch.nn.functional.normalize(bitangent, p=2, dim=1)  # (F, 3)

    # Initialize per-vertex tangent and bitangent accumulators
    V = vertices.shape[0]
    tangents = torch.zeros((V, 3), dtype=dtype, device=device)
    bitangents = torch.zeros((V, 3), dtype=dtype, device=device)

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


class OBJLoader(ModelLoader):
    def __init__(self, obj_file, texture_file=None, normal_file=None, metallic_roughness_file=None, emission_texture_file=None, flip_textures=(False, False, False, False)):
        super().__init__()
        self.obj_file = obj_file
        self.texture_file = texture_file
        self.normal_file = normal_file
        self.emission_texture_file = emission_texture_file
        self.metallic_roughness_file = metallic_roughness_file
        self.flip_textures = flip_textures

    def load(self):

        # Load .obj file using trimesh
        mesh = trimesh.load(self.obj_file, force='mesh')

        textures = None
        normals = None
        metallic_roughness = None
        emission_texture = None
        vertex_normals = None
        uv = None
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
        indices = torch.tensor(mesh.faces, dtype=torch.int32, device='cuda')

        has_vertex_normals = hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0
        if has_vertex_normals:
            vertex_normals = torch.tensor(mesh.vertex_normals , dtype=torch.float32, device='cuda')

        has_uv = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        if has_uv:
            uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device='cuda')


        if self.texture_file:
            texture_image = Image.open(self.texture_file).convert('RGBA')
            if self.flip_textures[0]:
                texture_image = texture_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            texture_data = np.array(texture_image).astype(np.uint8)
            texture_tensor = torch.tensor(texture_data, dtype=torch.uint8, device='cuda')
            textures = optix_renderer.TextureObject(texture_tensor)

        if self.normal_file:
            normal_image = Image.open(self.normal_file).convert('RGBA')
            if self.flip_textures[1]:
                normal_image = normal_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            normal_data = np.array(normal_image).astype(np.uint8)
            normal_tensor = torch.tensor(normal_data, dtype=torch.uint8, device='cuda')
            normals = optix_renderer.TextureObject(normal_tensor)

        if self.metallic_roughness_file:
            metallic_roughness_image = Image.open(self.metallic_roughness_file).convert('RGBA')
            if self.flip_textures[2]:
                metallic_roughness_image = metallic_roughness_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            metallic_roughness_data = np.array(metallic_roughness_image).astype(np.uint8)
            metallic_roughness_tensor = torch.tensor(metallic_roughness_data, dtype=torch.uint8, device='cuda')
            metallic_roughness = optix_renderer.TextureObject(metallic_roughness_tensor)

        if self.emission_texture_file:
            emission_texture_image = Image.open(self.emission_texture_file).convert('RGBA')
            if self.flip_textures[3]:
                emission_texture_image = emission_texture_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            emission_texture_data = np.array(emission_texture_image).astype(np.uint8)
            emission_texture_tensor = torch.tensor(emission_texture_data, dtype=torch.uint8, device='cuda')
            emission_texture = optix_renderer.TextureObject(emission_texture_tensor)

        if uv is not None:
            tangents, bitangents = compute_tangents(vertices, uv, indices, vertex_normals)
        else:
            # If UVs are not present, set tangents and bitangents to zero vectors
            tangents = torch.zeros_like(vertices, dtype=torch.float32, device='cuda')
            bitangents = torch.zeros_like(vertices, dtype=torch.float32, device='cuda')
        # Create Model object
        self.model = Model(vertices, indices, uv, tangents, bitangents, vertex_normals, textures, normals, metallic_roughness, emission_texture)
        return self.model


