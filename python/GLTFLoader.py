import trimesh
import torch
import numpy as np
from pygltflib import GLTF2
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt

def create_checkerboard_texture(size=512, num_squares=8):
    # Your checkerboard texture function remains unchanged
    square_size = size // num_squares
    checkerboard = torch.zeros((size, size, 4), dtype=torch.uint8)
    for y in range(size):
        for x in range(size):
            if ((x // square_size) % 2 == (y // square_size) % 2):
                checkerboard[y, x] = torch.tensor([255, 255, 255, 255], dtype=torch.uint8)
            else:
                checkerboard[y, x] = torch.tensor([0, 0, 0, 255], dtype=torch.uint8)
    return checkerboard.cuda()

class GLTFLoader:
    def __init__(self, gltf_dir, file):
        self.gltf_file_path = gltf_dir + "/" + file
        self.gltf_dir = gltf_dir
        self.vertices = []
        self.indices = []  # Separate list of indices for different meshes
        self.tex_coords = []
        self.textures = []
        self.mesh_texture_map = []  # Mapping from each mesh to its corresponding texture
        self.device = torch.device('cuda')
        self.node_matrices = {}  # Stores transformation matrices for each mesh

        self.load_gltf()
        self.apply_transformations()

    def load_gltf(self):
        # Load mesh data using trimesh
        scene = trimesh.load(self.gltf_file_path)
        for mesh_name, mesh in scene.geometry.items():
            # Extract vertices, indices, and texture coordinates
            vertices_tensor = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=self.device)
            indices_tensor = torch.tensor(np.array(mesh.faces), dtype=torch.int32, device=self.device)
            self.vertices.append(vertices_tensor)
            self.indices.append(indices_tensor)  # Store indices separately for each mesh

            if mesh.visual.uv is not None:
                tex_coords_tensor = torch.tensor(np.array(mesh.visual.uv), dtype=torch.float32, device=self.device)
                self.tex_coords.append(tex_coords_tensor)
            else:
                self.tex_coords.append(None)

        # Load texture data using pygltflib
        gltf = GLTF2().load(self.gltf_file_path)
        textures_dict = {}
        for image_index, image in enumerate(gltf.images):
            if image.uri.startswith('data:image'):
                # Handle embedded base64 texture
                header, encoded = image.uri.split(',', 1)
                image_data = base64.b64decode(encoded)
                image_file = io.BytesIO(image_data)
                texture_image = Image.open(image_file)
            else:
                # Handle texture from external file
                texture_image = Image.open(self.gltf_dir + "/" + image.uri)
            print(image.uri, texture_image)
            # Convert image to tensor and move to GPU
            texture_tensor = self.image_to_tensor(texture_image)
            self.textures.append(texture_tensor)
            textures_dict[image_index] = texture_tensor

        # Link meshes to textures using materials
        for mesh_index, mesh in enumerate(gltf.meshes):
            if len(mesh.primitives) > 0:
                material_index = mesh.primitives[0].material
                if material_index is not None and material_index < len(gltf.materials):
                    material = gltf.materials[material_index]
                    if material.pbrMetallicRoughness is not None and material.pbrMetallicRoughness.baseColorTexture is not None:
                        texture_index = material.pbrMetallicRoughness.baseColorTexture.index
                        if texture_index in textures_dict:
                            self.mesh_texture_map.append(textures_dict[texture_index])
                        else:
                            self.mesh_texture_map.append(None)
                    else:
                        self.mesh_texture_map.append(None)
                else:
                    self.mesh_texture_map.append(None)
            else:
                self.mesh_texture_map.append(None)

        # Extract node transformations
        for node_index, node in enumerate(gltf.nodes):
            if node.mesh is not None:
                translation = node.translation if node.translation else [0, 0, 0]
                rotation = node.rotation if node.rotation else [0, 0, 0, 1]  # [x, y, z, w] quaternion
                scale = node.scale if node.scale else [1, 1, 1]
                matrix = np.array(node.matrix).reshape((4, 4)) if node.matrix else self.get_transformation_matrix(translation, rotation, scale)
                self.node_matrices[node.mesh] = matrix

    def get_transformation_matrix(self, translation=None, rotation=None, scale=None):
        matrix = np.eye(4)

        # Apply translation
        if translation is not None:
            t = np.eye(4)
            t[:3, 3] = translation
            matrix = np.dot(matrix, t)

        # Apply rotation (Quaternion to rotation matrix)
        if rotation is not None:
            r = np.eye(4)
            q = rotation
            # Quaternion to rotation matrix
            r[:3, :3] = np.array([
                [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
                [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
                [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]
            ])
            matrix = np.dot(matrix, r)

        # Apply scale
        if scale is not None:
            s = np.eye(4)
            s[0, 0] = scale[0]
            s[1, 1] = scale[1]
            s[2, 2] = scale[2]
            matrix = np.dot(matrix, s)

        return matrix

    def get_translation_matrix_by_mesh_id(self, mesh_id):
        if mesh_id in self.node_matrices:
            return self.node_matrices[mesh_id]
        else:
            return None

    def apply_transformations(self):
        # Apply transformations to the model to ensure it's properly scaled, rotated, and centered.
        if len(self.vertices) == 0:
            return  # No vertices to transform

        all_vertices = torch.cat(self.vertices, dim=0)  # Combine all vertices
        min_coords = torch.min(all_vertices, dim=0).values
        max_coords = torch.max(all_vertices, dim=0).values
        center = (min_coords + max_coords) / 2.0

        # Translate all meshes to center around the origin
        for i in range(len(self.vertices)):
            self.vertices[i] -= center

        # Calculate the scaling factor to fit the model within a reasonable unit cube
        extent = max_coords - min_coords
        max_extent = torch.max(extent)
        scale_factor = 1.0 / max_extent * 4  # Scale to fit within [-1, 1] range

        # Apply scaling to all vertices
        for i in range(len(self.vertices)):
            self.vertices[i] *= scale_factor

    def image_to_tensor(self, image):
        # Ensure the image is in RGBA format (4 channels)
        image = image.convert('RGBA')

        # Convert image data to uint8 numpy array
        image_data = np.array(image).astype(np.uint8)  # Ensure the data type is uint8

        # Convert to a tensor and ensure it's on the GPU
        texture_tensor = torch.tensor(image_data, dtype=torch.uint8, device=self.device)

        return texture_tensor

    def get_vertices(self):
        return self.vertices

    def get_indices(self):
        return self.indices

    def get_tex_coords(self):
        return self.tex_coords

    def get_textures(self):
        return self.textures

    def get_texture_for_mesh(self, mesh_index):
        if 0 <= mesh_index < len(self.mesh_texture_map):
            return self.mesh_texture_map[mesh_index]
        return None

    def preview_textures(self):
        for i, texture in enumerate(self.textures):
            texture_np = texture.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) for visualization
            plt.figure()
            plt.imshow(texture_np)
            plt.title(f'Texture {i + 1}')
            plt.axis('off')
            plt.show()

if __name__ == '__main__':
    # Example usage
    rootPath = r"../models/elevator_building"
    gltf_loader = GLTFLoader(rootPath, 'scene.gltf')
    vertices_tensors = gltf_loader.get_vertices()
    indices_tensors = gltf_loader.get_indices()
    tex_coords_tensors = gltf_loader.get_tex_coords()
    textures = gltf_loader.get_textures()

    # Printing tensor information
    print("Number of Meshes Loaded:", len(vertices_tensors))
    for idx, vertices_tensor in enumerate(vertices_tensors):
        print(f"Mesh {idx}: Vertices Tensor:", vertices_tensor.shape)
        print(f"Mesh {idx}: Indices Tensor:", indices_tensors[idx].shape)
        if tex_coords_tensors[idx] is not None:
            print(f"Mesh {idx}: Texture Coordinates Tensor:", tex_coords_tensors[idx].shape)
        texture_tensor = gltf_loader.get_texture_for_mesh(idx)
        if texture_tensor is not None:
            print(f"Mesh {idx}: Associated Texture Shape:", texture_tensor.shape)

    # Example of getting the transformation matrix for a mesh by its ID
    mesh_id = 0  # Replace with your mesh ID
    transformation_matrix = gltf_loader.get_translation_matrix_by_mesh_id(mesh_id)
    if transformation_matrix is not None:
        print(f"Transformation Matrix for Mesh {mesh_id}:")
        print(transformation_matrix)
    else:
        print(f"No transformation matrix found for Mesh {mesh_id}")

    gltf_loader.preview_textures()
