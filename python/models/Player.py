from python.controllers import InputController
from python.models import Camera


class Player:
    def __init__(self, controller: InputController, camera: Camera):
        self.controller = controller
        self.camera = camera

    def update(self, dt):
        self.controller.update()
        movement = [i * dt for i in self.controller.get_movement()]
        rotation = [i * dt for i in self.controller.get_rotation()]
        self.camera.update(movement, rotation)
