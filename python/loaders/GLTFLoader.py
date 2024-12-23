import torch
import trimesh
import numpy as np
from pygltflib import GLTF2
from TextureLoader import TextureLoader
from Model import Model
from ModelLoader import ModelLoader

class GLTFLoader(ModelLoader):
    def __init__(self, gltf_dir, file):
        super().__init__()
        self.gltf_file_path = gltf_dir + "/" + file
        self.gltf_dir = gltf_dir
        self.device = torch.device('cuda')

    def load(self):
        # Load mesh data using trimesh
        scene = trimesh.load(self.gltf_file_path)
        vertices = []
        indices = []
        tex_coords = []
        textures = []

        for mesh_name, mesh in scene.geometry.items():
            vertices_tensor = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=self.device)
            indices_tensor = torch.tensor(np.array(mesh.faces), dtype=torch.int32, device=self.device)
            vertices.append(vertices_tensor)
            indices.append(indices_tensor)

            if mesh.visual.uv is not None:
                tex_coords_tensor = torch.tensor(np.array(mesh.visual.uv), dtype=torch.float32, device=self.device)
                tex_coords.append(tex_coords_tensor)
            else:
                tex_coords.append(None)

        # Load textures using pygltflib
        gltf = GLTF2().load(self.gltf_file_path)
        textures_dict = {}

        for image_index, image in enumerate(gltf.images):
            texture_loader = TextureLoader(self.gltf_dir, image.uri)
            texture_tensor = texture_loader.load_texture()
            textures.append(texture_tensor)
            textures_dict[image_index] = texture_tensor

        # Link meshes to textures using materials
        mesh_texture_map = []
        for mesh_index, mesh in enumerate(gltf.meshes):
            if len(mesh.primitives) > 0:
                material_index = mesh.primitives[0].material
                if material_index is not None and material_index < len(gltf.materials):
                    material = gltf.materials[material_index]
                    if material.pbrMetallicRoughness is not None and material.pbrMetallicRoughness.baseColorTexture is not None:
                        texture_index = material.pbrMetallicRoughness.baseColorTexture.index
                        if texture_index in textures_dict:
                            mesh_texture_map.append(textures_dict[texture_index])
                        else:
                            mesh_texture_map.append(None)
                    else:
                        mesh_texture_map.append(None)
                else:
                    mesh_texture_map.append(None)
            else:
                mesh_texture_map.append(None)

        # Create Model object
        self.model = Model(vertices, indices, tex_coords, textures)