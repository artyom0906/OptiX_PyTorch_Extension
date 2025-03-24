"""
Texture module for OptiX PyTorch Extension
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Any, Union, Tuple, List
import numpy as np
from enum import Enum

# Import texture types from the C++ extension
try:
    from optix_resource_system import TextureType
except ImportError:
    # Fallback enum for documentation/development without the extension
    class TextureType(Enum):
        RGB = 0
        RGBA = 1
        NORMAL_MAP = 2
        GRAYSCALE = 3
        HDR = 4


class Texture:
    """
    Texture class for creating and managing texture resources in the OptiX renderer.
    
    This class provides a convenient way to create and manipulate texture data
    before sending it to the GPU through the resource manager.
    """
    
    def __init__(self, data: Tensor, texture_type: TextureType = TextureType.RGB):
        """
        Initialize a texture object with tensor data.
        
        Args:
            data: Tensor containing texture data
            texture_type: The type of texture (RGB, RGBA, NORMAL_MAP, etc.)
        """
        # Ensure data is a tensor
        if not isinstance(data, Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Check dimensionality based on texture type
        if texture_type == TextureType.GRAYSCALE:
            # Grayscale textures can be 2D (H, W) or 3D (H, W, 1)
            if data.dim() == 2:
                # Already in correct format
                pass
            elif data.dim() == 3 and data.shape[2] == 1:
                # Already in correct format
                pass
            elif data.dim() == 3 and data.shape[2] >= 3:
                # Convert RGB to grayscale using standard weights
                data = 0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]
            else:
                raise ValueError(f"Invalid dimensions for grayscale texture: {data.shape}")
        elif texture_type in [TextureType.RGB, TextureType.NORMAL_MAP]:
            # RGB textures must be 3D (H, W, 3)
            if data.dim() == 3 and data.shape[2] == 3:
                # Already in correct format
                pass
            elif data.dim() == 3 and data.shape[2] == 4:
                # Extract RGB from RGBA
                data = data[..., :3]
            else:
                raise ValueError(f"Invalid dimensions for RGB texture: {data.shape}, expected (H, W, 3)")
        elif texture_type == TextureType.RGBA:
            # RGBA textures must be 3D (H, W, 4)
            if data.dim() == 3 and data.shape[2] == 4:
                # Already in correct format
                pass
            elif data.dim() == 3 and data.shape[2] == 3:
                # Add alpha channel
                h, w, _ = data.shape
                alpha = torch.ones((h, w, 1), dtype=data.dtype)
                data = torch.cat([data, alpha], dim=2)
            else:
                raise ValueError(f"Invalid dimensions for RGBA texture: {data.shape}, expected (H, W, 4)")
        elif texture_type == TextureType.HDR:
            # HDR textures are typically RGB or RGBA floating point
            if data.dim() == 3 and data.shape[2] in [3, 4]:
                # Already in correct format
                pass
            else:
                raise ValueError(f"Invalid dimensions for HDR texture: {data.shape}, expected (H, W, 3) or (H, W, 4)")
        
        # Make sure data is in float32
        if data.dtype != torch.float32:
            data = data.to(torch.float32)
        
        # Store texture data and type
        self.data = data
        self.texture_type = texture_type
    
    def create_gpu_resource(self, resource_manager) -> int:
        """
        Create a GPU resource from this texture.
        
        Args:
            resource_manager: ResourceManager instance to use for resource creation
            
        Returns:
            Handle to the created GPU resource
        """
        # CUDA arrays support 1, 2, and 4 channels but not 3 channels directly
        # Convert RGB to RGBA if needed, but keep the original texture type
        data = self.data
        if self.data.shape[-1] == 3:
            # RGB to RGBA conversion - add alpha channel
            h, w, _ = self.data.shape
            alpha = torch.ones((h, w, 1), dtype=self.data.dtype)
            data = torch.cat([self.data, alpha], dim=2)
            
        return resource_manager.create_texture(data, self.texture_type)
    
    @classmethod
    def from_file(cls, file_path: str, texture_type: Optional[TextureType] = None) -> 'Texture':
        """
        Create a texture from an image file.
        
        Args:
            file_path: Path to the image file
            texture_type: Optional texture type, if None it will be inferred from the image
            
        Returns:
            A new Texture instance
        """
        try:
            import imageio.v3 as iio
        except ImportError:
            raise ImportError("imageio is required to load image files. Install with: pip install imageio")
        
        # Load image data
        img = iio.imread(file_path)
        
        # Convert to tensor
        tensor_data = torch.from_numpy(img).to(torch.float32) / 255.0
        
        # Infer texture type if not specified
        if texture_type is None:
            if tensor_data.dim() == 2 or (tensor_data.dim() == 3 and tensor_data.shape[2] == 1):
                texture_type = TextureType.GRAYSCALE
            elif tensor_data.dim() == 3:
                if tensor_data.shape[2] == 3:
                    # Check if it's a normal map by filename
                    if "normal" in file_path.lower() or "nrm" in file_path.lower():
                        texture_type = TextureType.NORMAL_MAP
                    else:
                        texture_type = TextureType.RGB
                elif tensor_data.shape[2] == 4:
                    texture_type = TextureType.RGBA
                else:
                    raise ValueError(f"Unsupported number of channels: {tensor_data.shape[2]}")
            else:
                raise ValueError(f"Unsupported image dimensions: {tensor_data.shape}")
        
        return cls(tensor_data, texture_type)
    
    @classmethod
    def create_checkerboard(cls, width: int = 512, height: int = 512, 
                           color1: List[float] = [1.0, 1.0, 1.0], 
                           color2: List[float] = [0.0, 0.0, 0.0],
                           check_size: int = 64) -> 'Texture':
        """
        Create a checkerboard texture.
        
        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            color1: First color [r, g, b] in range [0, 1]
            color2: Second color [r, g, b] in range [0, 1]
            check_size: Size of each check in pixels
            
        Returns:
            A new Texture instance with a checkerboard pattern
        """
        # Create empty texture data
        data = torch.zeros((height, width, 3), dtype=torch.float32)
        
        # Convert colors to tensors
        color1 = torch.tensor(color1, dtype=torch.float32)
        color2 = torch.tensor(color2, dtype=torch.float32)
        
        # Generate checkerboard pattern
        for i in range(height):
            for j in range(width):
                if ((i // check_size) + (j // check_size)) % 2 == 0:
                    data[i, j] = color1
                else:
                    data[i, j] = color2
        
        return cls(data, TextureType.RGB)
    
    @classmethod
    def create_grid(cls, width: int = 512, height: int = 512, 
                   line_color: List[float] = [0.0, 0.0, 0.0],
                   bg_color: List[float] = [1.0, 1.0, 1.0],
                   line_width: int = 2,
                   grid_size: int = 64) -> 'Texture':
        """
        Create a grid texture.
        
        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            line_color: Grid line color [r, g, b] in range [0, 1]
            bg_color: Background color [r, g, b] in range [0, 1]
            line_width: Width of grid lines in pixels
            grid_size: Size of each grid cell in pixels
            
        Returns:
            A new Texture instance with a grid pattern
        """
        # Create texture with background color
        data = torch.zeros((height, width, 3), dtype=torch.float32)
        # Fill with background color
        bg_color_tensor = torch.tensor(bg_color, dtype=torch.float32)
        data[:, :] = bg_color_tensor
        
        # Convert colors to tensors
        line_color = torch.tensor(line_color, dtype=torch.float32)
        
        # Draw horizontal lines
        for i in range(0, height, grid_size):
            y_start = max(0, i - line_width // 2)
            y_end = min(height, i + line_width // 2 + 1)
            data[y_start:y_end, :] = line_color
        
        # Draw vertical lines
        for j in range(0, width, grid_size):
            x_start = max(0, j - line_width // 2)
            x_end = min(width, j + line_width // 2 + 1)
            data[:, x_start:x_end] = line_color
        
        return cls(data, TextureType.RGB)
    
    @classmethod
    def create_noise(cls, width: int = 512, height: int = 512, 
                    channels: int = 3, scale: float = 1.0, 
                    seed: Optional[int] = None) -> 'Texture':
        """
        Create a noise texture.
        
        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            channels: Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)
            scale: Scale factor for noise values
            seed: Random seed for reproducibility
            
        Returns:
            A new Texture instance with random noise
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Create random noise
        data = torch.rand((height, width, channels), dtype=torch.float32) * scale
        
        # Determine texture type
        if channels == 1:
            texture_type = TextureType.GRAYSCALE
        elif channels == 3:
            texture_type = TextureType.RGB
        elif channels == 4:
            texture_type = TextureType.RGBA
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
        
        return cls(data, texture_type)
    
    @classmethod
    def create_gradient(cls, width: int = 512, height: int = 512,
                       start_color: List[float] = [1.0, 0.0, 0.0],
                       end_color: List[float] = [0.0, 0.0, 1.0],
                       direction: str = 'horizontal') -> 'Texture':
        """
        Create a gradient texture.
        
        Args:
            width: Width of the texture in pixels
            height: Height of the texture in pixels
            start_color: Start color [r, g, b] in range [0, 1]
            end_color: End color [r, g, b] in range [0, 1]
            direction: Gradient direction ('horizontal', 'vertical', 'radial', or 'angular')
            
        Returns:
            A new Texture instance with a gradient pattern
        """
        # Convert colors to tensors
        start_color = torch.tensor(start_color, dtype=torch.float32)
        end_color = torch.tensor(end_color, dtype=torch.float32)
        
        # Create coordinates
        if direction == 'horizontal':
            # Generate values from 0 to 1 along the width
            t = torch.linspace(0, 1, width).view(1, width, 1).repeat(height, 1, 1)
        elif direction == 'vertical':
            # Generate values from 0 to 1 along the height
            t = torch.linspace(0, 1, height).view(height, 1, 1).repeat(1, width, 1)
        elif direction == 'radial':
            # Generate radial gradient from center
            x = torch.linspace(-1, 1, width).view(1, width)
            y = torch.linspace(-1, 1, height).view(height, 1)
            r = torch.sqrt(x.pow(2) + y.pow(2)) / torch.tensor(2).sqrt()
            t = torch.clamp(r, 0, 1).unsqueeze(2)
        elif direction == 'angular':
            # Generate angular gradient around center
            x = torch.linspace(-1, 1, width).view(1, width)
            y = torch.linspace(-1, 1, height).view(height, 1)
            angle = torch.atan2(y, x) / (2 * torch.tensor(np.pi))
            t = ((angle + 1) / 2).unsqueeze(2)
        else:
            raise ValueError(f"Unsupported gradient direction: {direction}")
        
        # Interpolate between colors
        data = start_color * (1 - t) + end_color * t
        
        return cls(data, TextureType.RGB)