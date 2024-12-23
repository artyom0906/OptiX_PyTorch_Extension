import trimesh
import torch
import numpy as np
from pygltflib import GLTF2
from PIL import Image
import base64
import io

class TextureLoader:
    def __init__(self, gltf_dir, texture_uri):
        self.gltf_dir = gltf_dir
        self.texture_uri = texture_uri
        self.texture_tensor = None

    def load_texture(self):
        if self.texture_uri.startswith('data:image'):
            # Handle embedded base64 texture
            header, encoded = self.texture_uri.split(',', 1)
            image_data = base64.b64decode(encoded)
            image_file = io.BytesIO(image_data)
            texture_image = Image.open(image_file)
        else:
            # Handle texture from external file
            texture_image = Image.open(self.gltf_dir + "/" + self.texture_uri)

        self.texture_tensor = self.image_to_tensor(texture_image)
        return self.texture_tensor

    def image_to_tensor(self, image):
        image = image.convert('RGBA')
        image_data = np.array(image).astype(np.uint8)
        return torch.tensor(image_data, dtype=torch.uint8)