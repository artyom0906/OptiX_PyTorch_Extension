"""
Geometry module for OptiX PyTorch Extension
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Any, Union, Tuple


class Geometry:
    """
    Geometry class for creating and managing geometry resources in the OptiX renderer.
    
    This class provides a convenient way to create and manipulate geometry data
    before sending it to the GPU through the resource manager.
    """
    
    def __init__(self, vertices: Tensor, indices: Optional[Tensor] = None):
        """
        Initialize a geometry object with vertex positions and optional indices.
        
        Args:
            vertices: Tensor of shape [N, 3] containing vertex positions
            indices: Optional tensor of shape [M, 3] containing triangle indices
        """
        # Ensure vertices are the right shape and type
        if not isinstance(vertices, Tensor):
            vertices = torch.tensor(vertices, dtype=torch.float32)
        if vertices.dim() != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Vertices must have shape [N, 3], got {vertices.shape}")
        if vertices.dtype != torch.float32:
            vertices = vertices.to(torch.float32)
        
        # Store vertices
        self.vertices = vertices
        
        # Process indices if provided
        if indices is not None:
            if not isinstance(indices, Tensor):
                indices = torch.tensor(indices, dtype=torch.int32)
            if indices.dim() == 2 and indices.shape[1] == 3:
                # Shape [M, 3] is correct
                pass
            elif indices.dim() == 1 and indices.numel() % 3 == 0:
                # Reshape flat indices to [M, 3]
                indices = indices.reshape(-1, 3)
            else:
                raise ValueError(f"Indices must have shape [M, 3] or [3*M], got {indices.shape}")
            if indices.dtype != torch.int32:
                indices = indices.to(torch.int32)
        
        self.indices = indices
        self.normals = None
        self.tex_coords = None
        self.tangents = None
        self.bitangents = None
    
    def set_normals(self, normals: Tensor) -> 'Geometry':
        """
        Set vertex normals.
        
        Args:
            normals: Tensor of shape [N, 3] containing normal vectors
            
        Returns:
            self for method chaining
        """
        if not isinstance(normals, Tensor):
            normals = torch.tensor(normals, dtype=torch.float32)
        if normals.dim() != 2 or normals.shape[1] != 3:
            raise ValueError(f"Normals must have shape [N, 3], got {normals.shape}")
        if normals.shape[0] != self.vertices.shape[0]:
            raise ValueError(f"Number of normals ({normals.shape[0]}) must match number of vertices ({self.vertices.shape[0]})")
        if normals.dtype != torch.float32:
            normals = normals.to(torch.float32)
        
        self.normals = normals
        return self
        
    def set_tangents(self, tangents: Tensor) -> 'Geometry':
        """
        Set vertex tangent vectors.
        
        Args:
            tangents: Tensor of shape [N, 3] containing tangent vectors
            
        Returns:
            self for method chaining
        """
        if not isinstance(tangents, Tensor):
            tangents = torch.tensor(tangents, dtype=torch.float32)
        if tangents.dim() != 2 or tangents.shape[1] != 3:
            raise ValueError(f"Tangents must have shape [N, 3], got {tangents.shape}")
        if tangents.shape[0] != self.vertices.shape[0]:
            raise ValueError(f"Number of tangents ({tangents.shape[0]}) must match number of vertices ({self.vertices.shape[0]})")
        if tangents.dtype != torch.float32:
            tangents = tangents.to(torch.float32)
        
        self.tangents = tangents
        return self
        
    def set_bitangents(self, bitangents: Tensor) -> 'Geometry':
        """
        Set vertex bitangent vectors.
        
        Args:
            bitangents: Tensor of shape [N, 3] containing bitangent vectors
            
        Returns:
            self for method chaining
        """
        if not isinstance(bitangents, Tensor):
            bitangents = torch.tensor(bitangents, dtype=torch.float32)
        if bitangents.dim() != 2 or bitangents.shape[1] != 3:
            raise ValueError(f"Bitangents must have shape [N, 3], got {bitangents.shape}")
        if bitangents.shape[0] != self.vertices.shape[0]:
            raise ValueError(f"Number of bitangents ({bitangents.shape[0]}) must match number of vertices ({self.vertices.shape[0]})")
        if bitangents.dtype != torch.float32:
            bitangents = bitangents.to(torch.float32)
        
        self.bitangents = bitangents
        return self
    
    def set_tex_coords(self, tex_coords: Tensor) -> 'Geometry':
        """
        Set texture coordinates.
        
        Args:
            tex_coords: Tensor of shape [N, 2] containing texture coordinates
            
        Returns:
            self for method chaining
        """
        if not isinstance(tex_coords, Tensor):
            tex_coords = torch.tensor(tex_coords, dtype=torch.float32)
        if tex_coords.dim() != 2 or tex_coords.shape[1] != 2:
            raise ValueError(f"Texture coordinates must have shape [N, 2], got {tex_coords.shape}")
        if tex_coords.shape[0] != self.vertices.shape[0]:
            raise ValueError(f"Number of texture coordinates ({tex_coords.shape[0]}) must match number of vertices ({self.vertices.shape[0]})")
        if tex_coords.dtype != torch.float32:
            tex_coords = tex_coords.to(torch.float32)
        
        self.tex_coords = tex_coords
        return self
    
    def compute_normals(self) -> 'Geometry':
        """
        Compute vertex normals based on triangle faces.
        
        Returns:
            self for method chaining
        """
        if self.indices is None:
            raise ValueError("Cannot compute normals without indices")
        
        # Get vertices for each triangle
        triangles = self.vertices[self.indices]
        
        # Compute face normals
        v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        face_normals = torch.nn.functional.normalize(torch.linalg.cross(v1 - v0, v2 - v0, dim=1), dim=1)
        
        # Initialize per-vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        
        # Accumulate face normals to vertices
        for i in range(self.indices.shape[0]):
            for j in range(3):
                vertex_idx = self.indices[i, j]
                vertex_normals[vertex_idx] += face_normals[i]
        
        # Normalize the results
        self.normals = torch.nn.functional.normalize(vertex_normals, dim=1)
        return self
        
    def create_gpu_resource(self, resource_manager) -> int:
        """
        Create a GPU resource from this geometry.
        
        Args:
            resource_manager: ResourceManager instance to use for resource creation
            
        Returns:
            Handle to the created GPU resource
        """
        # Create the geometry with all available attributes
        return resource_manager.create_geometry(
            vertices=self.vertices,
            indices=self.indices if self.indices is not None else None,
            normals=self.normals if self.normals is not None else None,
            tex_coords=self.tex_coords if self.tex_coords is not None else None,
            tangents=self.tangents if self.tangents is not None else None,
            bitangents=self.bitangents if self.bitangents is not None else None
        )
        
    def compute_tangent_space(self) -> 'Geometry':
        """
        Compute tangent space vectors based on texture coordinates.
        
        Returns:
            self for method chaining
        """
        if self.indices is None:
            raise ValueError("Cannot compute tangent space without indices")
        if self.tex_coords is None:
            raise ValueError("Cannot compute tangent space without texture coordinates")
        
        # Create empty tangent and bitangent arrays
        tangents = torch.zeros_like(self.vertices)
        bitangents = torch.zeros_like(self.vertices)
        
        # Get vertices and UVs for each triangle
        triangles = self.vertices[self.indices]
        uvs = self.tex_coords[self.indices]
        
        # For each triangle
        for i in range(self.indices.shape[0]):
            # Get triangle vertices and UVs
            v0, v1, v2 = triangles[i, 0], triangles[i, 1], triangles[i, 2]
            uv0, uv1, uv2 = uvs[i, 0], uvs[i, 1], uvs[i, 2]
            
            # Edge vectors
            e1 = v1 - v0
            e2 = v2 - v0
            
            # UV deltas
            delta_uv1 = uv1 - uv0
            delta_uv2 = uv2 - uv0
            
            # Calculate tangent and bitangent
            # This is solving the linear system:
            # e1 = delta_uv1.x * T + delta_uv1.y * B
            # e2 = delta_uv2.x * T + delta_uv2.y * B
            denom = delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0]
            if abs(denom) < 1e-6:
                # Handle degenerate case
                continue
                
            r = 1.0 / denom
            t = (delta_uv1[0] * e2 - delta_uv2[0] * e1) * r
            b = (delta_uv1[1] * e2 - delta_uv2[1] * e1) * r
            
            # Accumulate tangents for each vertex of the triangle
            for j in range(3):
                idx = self.indices[i, j]
                tangents[idx] += t
                bitangents[idx] += b
        
        # Normalize the results
        self.tangents = torch.nn.functional.normalize(tangents, dim=1)
        self.bitangents = torch.nn.functional.normalize(bitangents, dim=1)
        
        return self
    
    def create_cube(width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> 'Geometry':
        """
        Create a cube geometry.
        
        Args:
            width: Width of the cube (x-axis)
            height: Height of the cube (y-axis)
            depth: Depth of the cube (z-axis)
            
        Returns:
            A new Geometry instance representing a cube
        """
        w, h, d = width / 2, height / 2, depth / 2
        
        # Create 8 vertices
        vertices = torch.tensor([
            [-w, -h, -d],  # 0: left bottom back
            [+w, -h, -d],  # 1: right bottom back
            [+w, +h, -d],  # 2: right top back
            [-w, +h, -d],  # 3: left top back
            [-w, -h, +d],  # 4: left bottom front
            [+w, -h, +d],  # 5: right bottom front
            [+w, +h, +d],  # 6: right top front
            [-w, +h, +d],  # 7: left top front
        ], dtype=torch.float32)
        
        # Define 12 triangles (6 faces)
        indices = torch.tensor([
            # Back face
            [0, 1, 2], [0, 2, 3],
            # Front face
            [4, 6, 5], [4, 7, 6],
            # Left face
            [0, 3, 7], [0, 7, 4],
            # Right face
            [1, 5, 6], [1, 6, 2],
            # Bottom face
            [0, 4, 5], [0, 5, 1],
            # Top face
            [3, 2, 6], [3, 6, 7]
        ], dtype=torch.int32)
        
        return Geometry(vertices, indices)
    
    def create_plane(width: float = 1.0, height: float = 1.0, width_segments: int = 1, height_segments: int = 1) -> 'Geometry':
        """
        Create a plane geometry (grid).
        
        Args:
            width: Width of the plane (x-axis)
            height: Height of the plane (y-axis)
            width_segments: Number of segments in width direction
            height_segments: Number of segments in height direction
            
        Returns:
            A new Geometry instance representing a plane
        """
        # Ensure at least 1 segment
        width_segments = max(1, width_segments)
        height_segments = max(1, height_segments)
        
        # Calculate grid parameters
        width_half = width / 2
        height_half = height / 2
        grid_x = width_segments
        grid_y = height_segments
        segment_width = width / grid_x
        segment_height = height / grid_y
        
        # Create vertices
        vertices = []
        uvs = []
        
        for iy in range(grid_y + 1):
            y = iy * segment_height - height_half
            for ix in range(grid_x + 1):
                x = ix * segment_width - width_half
                vertices.append([x, 0, y])  # Using Y-up in world space
                uvs.append([ix / grid_x, 1 - (iy / grid_y)])  # V should be flipped
        
        vertices = torch.tensor(vertices, dtype=torch.float32)
        uvs = torch.tensor(uvs, dtype=torch.float32)
        
        # Create indices
        indices = []
        for iy in range(grid_y):
            for ix in range(grid_x):
                a = (grid_x + 1) * iy + ix
                b = (grid_x + 1) * (iy + 1) + ix
                c = (grid_x + 1) * (iy + 1) + (ix + 1)
                d = (grid_x + 1) * iy + (ix + 1)
                
                # Two triangles per grid cell
                indices.append([a, b, d])
                indices.append([b, c, d])
        
        indices = torch.tensor(indices, dtype=torch.int32)
        
        # Create geometry and set texture coordinates
        geo = Geometry(vertices, indices)
        geo.set_tex_coords(uvs)
        geo.set_normals(torch.tensor([[0, 1, 0]] * vertices.shape[0], dtype=torch.float32))
        
        return geo
    
    def create_sphere(radius: float = 1.0, width_segments: int = 32, height_segments: int = 16) -> 'Geometry':
        """
        Create a sphere geometry.
        
        Args:
            radius: Radius of the sphere
            width_segments: Number of segments around the equator
            height_segments: Number of segments from pole to pole
            
        Returns:
            A new Geometry instance representing a sphere
        """
        # Ensure at least 3 segments
        width_segments = max(3, width_segments)
        height_segments = max(2, height_segments)
        
        # Calculate grid parameters
        grid_x = width_segments
        grid_y = height_segments
        
        # Create vertices
        vertices = []
        uvs = []
        normals = []
        
        for iy in range(grid_y + 1):
            v = iy / grid_y
            phi = v * 3.14159  # 0 to pi (top to bottom)
            
            for ix in range(grid_x + 1):
                u = ix / grid_x
                theta = u * 2 * 3.14159  # 0 to 2pi (around equator)
                
                # Calculate vertex position using math
                import math
                x = -radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                
                vertices.append([x, y, z])
                uvs.append([u, 1 - v])  # V should be flipped
                normals.append([x/radius, y/radius, z/radius])  # Normals point outward
        
        vertices = torch.tensor(vertices, dtype=torch.float32)
        uvs = torch.tensor(uvs, dtype=torch.float32)
        normals = torch.tensor(normals, dtype=torch.float32)
        
        # Create indices
        indices = []
        for iy in range(grid_y):
            for ix in range(grid_x):
                a = (grid_x + 1) * iy + ix
                b = (grid_x + 1) * (iy + 1) + ix
                c = (grid_x + 1) * (iy + 1) + (ix + 1)
                d = (grid_x + 1) * iy + (ix + 1)
                
                # Two triangles per grid cell, skip degenerate triangles at poles
                if iy != 0:
                    indices.append([a, b, d])
                if iy != grid_y - 1:
                    indices.append([b, c, d])
        
        indices = torch.tensor(indices, dtype=torch.int32)
        
        # Create geometry and set attributes
        geo = Geometry(vertices, indices)
        geo.set_tex_coords(uvs)
        geo.set_normals(normals)
        
        return geo