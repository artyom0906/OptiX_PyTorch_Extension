import math
from gettext import translation

import numpy as np
import torch
import copy


class Animator:
    def __init__(self, instances: list, transformation, duration):
        self.transform_matrix = None
        self.instances = [(i, torch.eye(4, dtype=torch.float32).cuda()) for i in instances]
        self.transformation = transformation
        self.duration = duration
        self.elapsed = 0
        self.is_complete = False


    def update(self, dt):
        if self.is_complete:
            return
        self.elapsed += dt
        progress = min(self.elapsed / self.duration, 1.0)

        for i, mat in self.instances:
            #print(i, mat)
            i.set_transform_matrix(self.transformation(mat, progress))
            torch.eye(4, out=mat)

        if self.elapsed >= self.duration:
            self.is_complete = True
def lerp(start, end, t):
    return start + (end - start) * t
class Transformation:
    def __init__(self):
        self.next = []

    def call(self, mat, progress):
        return mat
    def __call__(self, mat, progress):
        mat = self.call(mat, progress)
        for t in self.next:
            mat = t(mat, progress)
        return mat

    def __mul__(self, other):
        if isinstance(other, Transformation):
            self.next.append(other)
        #print("mul", self.next)
        return self

class Translate(Transformation):
    def __init__(self, begin, end):
        super().__init__()
        self.begin = np.array(begin)
        self.end = np.array(end)

    def call(self, mat, progress):
        translation = lerp(self.begin, self.end, progress)
        mat[:3, 3] = torch.tensor(translation, dtype=torch.float32).cuda()
        #print("translate", progress, mat, self.begin, self.end)
        return mat

class Rotate(Transformation):
    def __init__(self, begin, end):
        super().__init__()
        self.begin = np.array(begin) * math.pi/180
        self.end = np.array(end) * math.pi/180

    def call(self, mat, progress):
        rotation = lerp(self.begin, self.end, progress)
        #rotation *= math.pi/180
        print((rotation[1]/math.pi)*180)
        rotation_matrix = self.apply_rotation(rotation.tolist())
        #print("rotate", mat, self.begin, self.end)
        # Apply rotation to the transformation matrix
        mat[:3, :3] = torch.matmul(mat[:3, :3], rotation_matrix)
        return mat

    def apply_rotation(self, rot):
        """
        Apply the rotation to the given matrix using Euler angles.
        """
        cos_x, sin_x = math.cos(rot[0]), math.sin(rot[0])
        cos_y, sin_y = math.cos(rot[1]), math.sin(rot[1])
        cos_z, sin_z = math.cos(rot[2]), math.sin(rot[2])

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
        return rotation_matrix


class filler:
    def __init__(self, mat):
        self.mat = mat
    def get_transformation_matrix_val(self):
        return self.mat

    def set_transform_matrix(self, mat):
        return mat

if __name__ == '__main__':
    matrix = torch.eye(4, dtype=torch.float32).cuda()
    transform = Translate([0, 0, 0], [30, 90, 40]) * Rotate([0, 0, 0], [0, 90, 0])
    animator = Animator([filler(matrix)], transform, 3000)

    for i in range(31):
        animator.update(100)
    print(animator.is_complete, matrix)