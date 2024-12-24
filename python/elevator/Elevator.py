import torch

from python.animation import Animator
from python.animation.animation import Translate
from python.elevator.AnimatableObject import AnimatableObjectBuilder


class ElevatorBuilder:
    def __init__(self, renderer, walls_model, floor_ceiling_model, doors_builder):
        self.walls_builder = AnimatableObjectBuilder(renderer, walls_model)
        self.floor_ceiling_builder = AnimatableObjectBuilder(renderer, floor_ceiling_model)
        self.doors_builder = doors_builder

    def build(self, translation, rotation, scale):
        return Elevator(self.walls_builder.build(translation, rotation, scale, is_glass=True),
                        self.floor_ceiling_builder.build(translation, rotation, scale, emission=(0, 5, 5)),
                        [builder.build(translation, rotation, scale) for builder in self.doors_builder])
class Elevator:
    def __init__(self, walls, floor_ceiling, doors):
        self.walls = walls
        self.floor_ceiling = floor_ceiling
        self.doors = doors
        self.floors = []
        self.current_floor = 0

    def add_floor(self, floor):
            self.floors.append(floor)

    def next_floor(self):
        next_floor = (self.current_floor + 1) % len(self.floors)
        c, n = self.floors[self.current_floor], self.floors[next_floor]
        direction = [n[i]-c[i] for i in range(3)]
        for door in self.doors:
            initial_transform = torch.mul(door.animatable_object.initial_transform, door.transform_close(door.animatable_object.initial_transform, 1.0))
            door.set_animation(Translate([0, 0, 0], direction), 5000)
        self.walls.set_animation(Translate([0, 0, 0], direction), 5000)
        self.floor_ceiling.set_animation(Translate([0, 0, 0], direction), 5000)
        self.current_floor = next_floor

    def open_doors(self):
            for door in self.doors:
                door.open()

    def close_doors(self):
        for door in self.doors:
            door.close()

    def update(self, dt):
        updated = False
        updated = self.walls.update(dt) or updated
        updated = self.floor_ceiling.update(dt) or updated
        for door in self.doors:
            updated = door.update(dt) or updated
        return updated

    def reset_position(self):
        self.walls.reset_position()
        self.floor_ceiling.reset_position()
        for door in self.doors:
            door.reset_position()
