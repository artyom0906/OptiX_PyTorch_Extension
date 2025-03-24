from python.controllers import InputController
from python.models import Camera


class Player:
    def __init__(self, controller: InputController, cameras: [Camera]):
        self.controller = controller
        self.cameras = cameras

    def update(self, dt):
        self.controller.update()
        movement = [i * dt for i in self.controller.get_movement()]
        rotation = [i * dt for i in self.controller.get_rotation()]
        for camera in self.cameras:
            camera.update(movement, rotation)
