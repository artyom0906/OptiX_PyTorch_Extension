import torch
import math

from python.loaders.ModelInstance import ModelInstance


class Model:
    def __init__(self, vertices, indices = None, tex_coords=None, tangents=None, bitangents=None, vertex_normals=None,
                 textures=None, normals=None, metallic_roughness = None, emission_texture = None):
        self.vertices = vertices  # List of vertex tensors
        self.indices = indices  # List of index tensors (can be None)
        self.tangents = tangents
        self.bitangents = bitangents
        self.vertex_normals = vertex_normals
        self.tex_coords = tex_coords  # List of texture coordinates (can be None)
        self.textures = textures  # List of texture tensors (can be None)
        self.normals = normals # List of normals tensors (can be None)
        self.metallic_roughness = metallic_roughness
        self.emission_texture   = emission_texture

    def get_geometry(self):
        """
        Return the geometry (vertices, indices, normals, and texture coordinates) of the model.
        """
        return self.vertices, self.indices, self.tex_coords, self.tangents, self.bitangents, self.vertex_normals

    def get_textures(self):
        """
        Return the textures of the model.
        """
        return self.textures, self.normals

    def create_geometry(self, renderer):
        """
        Create the geometry object for the renderer and return the geometry ID.
        This will be used by the ModelInstance.
        """
        # Get the model's geometry (vertices, indices, and texture coordinates)
        vertices, indices, tex_coords, tangents, bitangents, vertex_normals = self.get_geometry()

        # You might need to convert tex_coords or other parts depending on renderer requirements
        #if tex_coords is None:
        #    tex_coords = [None] * len(vertices)  # Fallback to None if no texture coordinates

        # Create and return the geometry using the renderer's function
        geometry = renderer.createVertexGeometry(vertices, indices, tex_coords, tangents, bitangents, vertex_normals, self.textures, self.normals, self.metallic_roughness, self.emission_texture)
        return geometry

    def add(self, renderer, color=(1, 1, 1), texture=None, is_glass=False, emission=(0,0,0)):
        """
        Add the model to the renderer and return a ModelInstance that manages its transformations.
        """
        # Create geometry for the model
        geometry = self.create_geometry(renderer)
        geometry.setMaterialColor(*color)
        geometry.setGlass(is_glass)
        geometry.setEmission(*emission)
        if texture:
            geometry.setTexture(texture)

        # Add geometry to the renderer and get the geometry instance ID
        geometry_id = renderer.addGeometryInstance(geometry)

        # Create a ModelInstance to handle transformations and GPU matrix
        model_instance = ModelInstance(self, geometry_id, renderer)

        return model_instance
