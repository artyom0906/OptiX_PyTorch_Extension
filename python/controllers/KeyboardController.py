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
        # Increase sensitivity for better responsiveness
        self.move_x = (keys[pygame.K_a] - keys[pygame.K_d]) * 0.2
        self.move_y = (keys[pygame.K_w] - keys[pygame.K_s]) * 0.2
        
        # Flip the vertical movement controls to match flipped coordinate system
        # Now E is down and Q is up (opposite of before)
        self.move_z = (keys[pygame.K_q] - keys[pygame.K_e]) * 0.2
        
        self.rotate = (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]) * 0.5
        
        # Flip pitch controls - now DOWN is up and UP is down
        self.pitch = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * 0.5

    def get_movement(self):
        return self.move_x, self.move_y, self.move_z

    def get_rotation(self):
        return self.rotate, self.pitch
