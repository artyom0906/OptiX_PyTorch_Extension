"""
OptiX PyTorch Extension - Resource System

This package provides GPU-accelerated ray tracing capabilities for PyTorch
using NVIDIA's OptiX ray tracing engine.
"""

from .geometry import Geometry
from .texture import Texture
from .material import Material

__all__ = ['Geometry', 'Texture', 'Material']