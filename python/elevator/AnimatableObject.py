import torch

from python.animation.animation import Transformation, Animator
from python.loaders.ModelInstance import apply_rotation


class AnimatableObjectBuilder:
    def __init__(self, renderer, model):
        self.model = model
        self.renderer = renderer

    def build(self, translation, rotation, scale, is_glass=False, emission=(0, 0, 0)):
        return AnimatableObject(self.renderer, self.model.add(self.renderer, is_glass=is_glass, emission=emission), build_transform_matrix(translation, rotation, scale))

def build_transform_matrix(translation, rotation, scale):
    transform_matrix = torch.eye(4, dtype=torch.float32).cuda()

    if translation:
        transform_matrix[:3, 3] = torch.tensor(translation, dtype=torch.float32).cuda()

    if rotation:
        rot_x, rot_y, rot_z = rotation
        transform_matrix = apply_rotation(transform_matrix, rot_x, rot_y, rot_z)

    if scale:
        transform_matrix[:3, :3] *= torch.tensor(scale, dtype=torch.float32).cuda()

    return transform_matrix
class AnimatableObject:
    def __init__(self, renderer, instance, initial_transform):
        self.instance = instance
        self.initial_transform = initial_transform.detach().clone()
        self.animator = Animator([(self.instance, initial_transform)], Transformation(), 1)
        self.current_transform = self.initial_transform

    def update(self, dt):
        if not self.animator.is_complete:
            self.current_transform = self.animator.update_mat(dt)[0]
        else:
            return False
        return True


    def set_animation(self, transform, duration):
        self.animator = Animator([(self.instance, self.current_transform)], transform, duration)

    def set_animator(self, animator):
        self.animator = animator


    def reset_position(self):
        self.instance.reset_transform_matrix()
        self.animator = Animator([(self.instance, self.initial_transform)], Transformation(), 1)