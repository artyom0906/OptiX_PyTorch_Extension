import pygame

from python.controllers.InputController import InputController
class KeyboardController(InputController):
    def __init__(self):
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_z = 0.0
        self.rotate = 0.0
        self.pitch = 0.0

    def update(self):
        keys = pygame.key.get_pressed()
        self.move_x = (keys[pygame.K_d] - keys[pygame.K_a]) * 0.1
        self.move_y = (keys[pygame.K_w] - keys[pygame.K_s]) * 0.1
        self.move_z = (keys[pygame.K_e] - keys[pygame.K_q]) * 0.1
        self.rotate = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 0.01
        self.pitch = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 0.01

    def get_movement(self):
        return self.move_x, self.move_y, self.move_z

    def get_rotation(self):
        return self.rotate, self.pitch
