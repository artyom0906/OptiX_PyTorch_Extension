"""
Material module for OptiX PyTorch Extension
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Any, Union, Tuple, List
from enum import Enum

# Import material types from the C++ extension
try:
    from optix_resource_system import MaterialType
except ImportError:
    # Fallback enum for documentation/development without the extension
    class MaterialType(Enum):
        LAMBERTIAN = 0
        PBR = 1
        GLASS = 2
        EMISSIVE = 3
        MIRROR = 4

# Import Texture class
try:
    from .texture import Texture  # Package import when installed
except ImportError:
    from texture import Texture  # Direct import for development


class Material:
    """
    Material class for creating and managing material resources in the OptiX renderer.
    
    This class provides a convenient way to create and manipulate material parameters
    before sending them to the GPU through the resource manager.
    """
    
    def __init__(self, material_type: MaterialType):
        """
        Initialize a material with the specified type.
        
        Args:
            material_type: The type of material (LAMBERTIAN, PBR, etc.)
        """
        self.material_type = material_type
        self.parameters = {}
        self.textures = {}
        
        # Initialize default parameters based on material type
        self._init_default_parameters()
    
    def _init_default_parameters(self):
        """Initialize default parameters based on material type"""
        if self.material_type == MaterialType.LAMBERTIAN:
            self.parameters['albedo'] = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32)
            
        elif self.material_type == MaterialType.PBR:
            self.parameters['base_color'] = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32)
            self.parameters['metallic'] = torch.tensor(0.0, dtype=torch.float32)
            self.parameters['roughness'] = torch.tensor(0.5, dtype=torch.float32)
            self.parameters['ior'] = torch.tensor(1.5, dtype=torch.float32)
            
        elif self.material_type == MaterialType.GLASS:
            self.parameters['transmittance'] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            self.parameters['ior'] = torch.tensor(1.5, dtype=torch.float32)
            self.parameters['roughness'] = torch.tensor(0.0, dtype=torch.float32)
            
        elif self.material_type == MaterialType.EMISSIVE:
            self.parameters['emission_color'] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            self.parameters['emission_intensity'] = torch.tensor(1.0, dtype=torch.float32)
            
        elif self.material_type == MaterialType.MIRROR:
            self.parameters['reflectivity'] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            self.parameters['roughness'] = torch.tensor(0.0, dtype=torch.float32)
    
    def set_parameter(self, name: str, value: Union[float, List[float], Tensor]) -> 'Material':
        """
        Set a material parameter.
        
        Args:
            name: Name of the parameter
            value: Value to set (float, list of floats, or tensor)
            
        Returns:
            self for method chaining
        """
        # Convert value to tensor if needed
        if not isinstance(value, Tensor):
            if isinstance(value, (list, tuple)):
                value = torch.tensor(value, dtype=torch.float32)
            else:
                value = torch.tensor(float(value), dtype=torch.float32)
        
        # Ensure value is float32
        if value.dtype != torch.float32:
            value = value.to(torch.float32)
        
        # Store the parameter
        self.parameters[name] = value
        return self
    
    def set_texture(self, name: str, texture: Texture) -> 'Material':
        """
        Set a material texture.
        
        Args:
            name: Name of the texture parameter
            texture: Texture object
            
        Returns:
            self for method chaining
        """
        self.textures[name] = texture
        return self
    
    def create_gpu_resource(self, resource_manager) -> int:
        """
        Create a GPU resource from this material.
        
        Args:
            resource_manager: ResourceManager instance to use for resource creation
            
        Returns:
            Handle to the created GPU resource
        """
        # First create the base material
        material_handle = resource_manager.create_material(self.material_type)
        
        # TODO: Set parameters and textures once supported by the resource manager
        
        return material_handle
    
    @classmethod
    def create_lambertian(cls, albedo: Union[List[float], Tensor] = [0.8, 0.8, 0.8]) -> 'Material':
        """
        Create a lambertian (diffuse) material.
        
        Args:
            albedo: Diffuse color (RGB values in range [0,1])
            
        Returns:
            A new Material instance
        """
        material = cls(MaterialType.LAMBERTIAN)
        material.set_parameter('albedo', albedo)
        return material
    
    @classmethod
    def create_pbr(cls, 
                  base_color: Union[List[float], Tensor] = [0.8, 0.8, 0.8],
                  metallic: float = 0.0,
                  roughness: float = 0.5,
                  ior: float = 1.5) -> 'Material':
        """
        Create a physically-based rendering (PBR) material.
        
        Args:
            base_color: Base color (RGB values in range [0,1])
            metallic: Metallic factor (0 = dielectric, 1 = metal)
            roughness: Surface roughness (0 = smooth, 1 = rough)
            ior: Index of refraction
            
        Returns:
            A new Material instance
        """
        material = cls(MaterialType.PBR)
        material.set_parameter('base_color', base_color)
        material.set_parameter('metallic', metallic)
        material.set_parameter('roughness', roughness)
        material.set_parameter('ior', ior)
        return material
    
    @classmethod
    def create_glass(cls, 
                    transmittance: Union[List[float], Tensor] = [1.0, 1.0, 1.0],
                    ior: float = 1.5,
                    roughness: float = 0.0) -> 'Material':
        """
        Create a glass material.
        
        Args:
            transmittance: Transmission color (RGB values in range [0,1])
            ior: Index of refraction
            roughness: Surface roughness (0 = smooth, 1 = rough)
            
        Returns:
            A new Material instance
        """
        material = cls(MaterialType.GLASS)
        material.set_parameter('transmittance', transmittance)
        material.set_parameter('ior', ior)
        material.set_parameter('roughness', roughness)
        return material
    
    @classmethod
    def create_emissive(cls, 
                       emission_color: Union[List[float], Tensor] = [1.0, 1.0, 1.0],
                       emission_intensity: float = 1.0) -> 'Material':
        """
        Create an emissive material.
        
        Args:
            emission_color: Emission color (RGB values in range [0,1])
            emission_intensity: Emission intensity multiplier
            
        Returns:
            A new Material instance
        """
        material = cls(MaterialType.EMISSIVE)
        material.set_parameter('emission_color', emission_color)
        material.set_parameter('emission_intensity', emission_intensity)
        return material
    
    @classmethod
    def create_mirror(cls, 
                     reflectivity: Union[List[float], Tensor] = [1.0, 1.0, 1.0],
                     roughness: float = 0.0) -> 'Material':
        """
        Create a mirror material.
        
        Args:
            reflectivity: Reflection color (RGB values in range [0,1])
            roughness: Surface roughness (0 = smooth, 1 = rough)
            
        Returns:
            A new Material instance
        """
        material = cls(MaterialType.MIRROR)
        material.set_parameter('reflectivity', reflectivity)
        material.set_parameter('roughness', roughness)
        return material