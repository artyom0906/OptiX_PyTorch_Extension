import torch
import math


def apply_rotation(matrix, rot_x, rot_y, rot_z):
    """
    Apply the rotation to the given matrix using Euler angles.
    """
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_z, sin_z = math.cos(rot_z), math.sin(rot_z)

    # Rotation matrices for X, Y, and Z axes
    rot_x_matrix = torch.tensor([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ]).cuda()

    rot_y_matrix = torch.tensor([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ]).cuda()

    rot_z_matrix = torch.tensor([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ]).cuda()

    # Combine rotations
    rotation_matrix = torch.matmul(rot_z_matrix, torch.matmul(rot_y_matrix, rot_x_matrix))

    # Apply rotation to the transformation matrix
    matrix[:3, :3] = torch.matmul(matrix[:3, :3], rotation_matrix)
    return matrix


class ModelInstance:
    def __init__(self, model, geometry_id, renderer):
        """
        Initialize the ModelInstance with a reference to the Model, a geometry ID, and a renderer.
        This will manage transformations for the model instance on the GPU.
        """
        self.model = model  # Reference to the Model object
        self.geometry_id = geometry_id  # ID for this geometry instance in the renderer
        self.renderer = renderer  # The renderer responsible for handling this instance
        self.transform_matrix = None

    def reset_transform_matrix(self):
        self.transform_matrix = None

    def set_transform(self, translation=None, rotation=None, scale=None):
        """
        Set a new transformation for the model instance (translation, rotation, scale).
        All arguments are optional.
        """
        # Start with the identity matrix

        if self.transform_matrix is None:
            self.transform_matrix = self.renderer.getTransformForInstance(self.geometry_id - 1)  # GPU-side transformation matrix

        transform_matrix = torch.eye(4, dtype=torch.float32).cuda()

        if translation:
            transform_matrix[:3, 3] = torch.tensor(translation, dtype=torch.float32).cuda()

        if rotation:
            rot_x, rot_y, rot_z = rotation
            transform_matrix = apply_rotation(transform_matrix, rot_x, rot_y, rot_z)

        if scale:
            transform_matrix[:3, :3] *= torch.tensor(scale, dtype=torch.float32).cuda()

        # Apply the new transformation directly to the GPU transform matrix for this geometry instance
        self.transform_matrix[:, :4] = transform_matrix[:3, :4]

    def add_transform(self, translation=None, rotation=None, scale=None):
        """
        Add a transformation to the model instance (translation, rotation, scale).
        All arguments are optional. The new transformation is combined with the existing one.
        """
        # Create a new transform matrix for the additional transformation
        if self.transform_matrix is None:
            self.transform_matrix = self.renderer.getTransformForInstance(self.geometry_id - 1)  # GPU-side transformation matrix
        transform_matrix = torch.eye(4, dtype=torch.float32).cuda()

        if translation:
            transform_matrix[:3, 3] = torch.tensor(translation, dtype=torch.float32).cuda()

        if rotation:
            rot_x, rot_y, rot_z = rotation
            transform_matrix = apply_rotation(transform_matrix, rot_x, rot_y, rot_z)

        if scale:
            transform_matrix[:3, :3] *= torch.tensor(scale, dtype=torch.float32).cuda()

        # Combine the current transformation with the new one
        self.transform_matrix = torch.matmul(self.transform_matrix, transform_matrix)

        # Apply the updated transformation directly to the GPU transform matrix for this geometry instance
        self.renderer.getTransformForInstance(self.geometry_id - 1)[:] = self.transform_matrix

    def set_transform_matrix(self, matrix):
        if self.transform_matrix is None:
            self.transform_matrix = self.renderer.getTransformForInstance(self.geometry_id - 1)
        self.transform_matrix[:, :4] = matrix[:3, :4]

    def get_transformation_matrix_val(self):
        if self.transform_matrix is None:
            self.transform_matrix = self.renderer.getTransformForInstance(self.geometry_id - 1)  # GPU-side transformation matrix
        transform_matrix = torch.eye(4, dtype=torch.float32).cuda()
        transform_matrix[:3, :4] = self.transform_matrix[:3, :4]
        return transform_matrix

