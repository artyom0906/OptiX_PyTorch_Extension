import pygame

from python.controllers.InputController import InputController

class GamepadController(InputController):
    def __init__(self):
        pygame.joystick.init()
        self.joysticks = []
        for i in range(pygame.joystick.get_count()):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            self.joysticks.append(joystick)

        # Sensitivity and dead zone
        self.DEAD_ZONE = 0.1
        self.movement_speed = 10
        self.rotation_speed = 1

        # Input states
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_z = 0.0
        self.rotate = 0.0
        self.pitch = 0.0

    def apply_dead_zone(self, value):
        return 0.0 if abs(value) < self.DEAD_ZONE else value

    def update(self):
        # Reset inputs
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_z = 0.0
        self.rotate = 0.0
        #self.pitch = 0.0

        if len(self.joysticks) > 0:
            joystick = self.joysticks[0]

            # Read raw axis values
            raw_move_x = -joystick.get_axis(0)  # Left stick horizontal
            raw_move_y = joystick.get_axis(1)  # Left stick vertical
            raw_move_z = joystick.get_axis(3)  # Right stick vertical (optional)
            raw_rotate = -joystick.get_axis(2)  # Right stick horizontal
            #raw_pitch = -joystick.get_axis(5)  # Adjust based on controller layout

            #print("ms", self.movement_speed)
            # Apply dead zone and normalize inputs
            self.move_x = self.apply_dead_zone(raw_move_x) * self.movement_speed
            self.move_y = self.apply_dead_zone(raw_move_y) * self.movement_speed
            self.move_z = self.apply_dead_zone(raw_move_z) * self.movement_speed
            self.rotate = self.apply_dead_zone(raw_rotate) * self.rotation_speed
            #self.pitch = self.apply_dead_zone(raw_pitch) * self.rotation_speed
            #print(self.move_x, self.move_y, self.move_z)

    def get_movement(self):
        return self.move_x, self.move_y, self.move_z

    def get_rotation(self):
        return self.rotate, self.pitch
