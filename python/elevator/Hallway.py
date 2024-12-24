from python.elevator.AnimatableObject import AnimatableObjectBuilder
class HallwayBuilder:
    def __init__(self, renderer, model, doors_builder):
        self.hallway_builder = AnimatableObjectBuilder(renderer, model)
        self.doors_builder = doors_builder

    def build(self, translation, rotation, scale):
        return Hallway(self.hallway_builder.build(translation, rotation, scale),
                        [builder.build(translation, rotation, scale) for builder in self.doors_builder])
class Hallway:
    def __init__(self, hallway, doors):
        self.hallway = hallway
        self.doors = doors

    def open_doors(self):
        for door in self.doors:
            door.open()

    def close_doors(self):
        for door in self.doors:
            door.close()

    def update(self, dt):
        updated = False
        updated = self.hallway.update(dt) or updated
        for door in self.doors:
            updated = door.update(dt) or updated
        return updated

    def reset_position(self):
        self.hallway.reset_position()
        for door in self.doors:
            door.reset_position()
