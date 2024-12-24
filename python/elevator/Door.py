import torch

from python.animation import Animator
from python.animation.animation import Transformation
from python.elevator.AnimatableObject import AnimatableObjectBuilder, AnimatableObject
from python.loaders.ModelInstance import apply_rotation


class DoorBuilder(AnimatableObjectBuilder):
    def __init__(self, renderer, model, transform_open, transform_close):
        super().__init__(renderer, model)
        self.transform_open = transform_open
        self.transform_close = transform_close

    def build(self, translation, rotation, scale):
        return Door(super().build(translation, rotation, scale), self.transform_open, self.transform_close)

class Door:
    def __init__(self, animatable_object, transform_open, transform_close):
        self.animatable_object = animatable_object
        self.transform_open = transform_open
        self.transform_close = transform_close


    def set_animation(self, transform, duration):
        self.animatable_object.set_animation(transform, duration)

    def update(self, dt):
        return self.animatable_object.update(dt)

    def reset_position(self):
        self.animatable_object.reset_position()
    def open(self):
        self.animatable_object.set_animation(self.transform_open, 3000)

    def close(self):
        self.animatable_object.set_animation(self.transform_close, 3000)

