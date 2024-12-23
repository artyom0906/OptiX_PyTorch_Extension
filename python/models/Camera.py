import math

import numpy as np
import torch


class Camera:
    def __init__(self, position, lookat, up):
        self.position = torch.tensor(position, dtype=torch.float32, device='cuda')
        self.lookat = torch.tensor(lookat, dtype=torch.float32, device='cuda')
        self.up = torch.tensor(up, dtype=torch.float32, device='cuda')
        self.yaw = 0.0
        self.pitch = 0.0

    def update(self, movement, rotation):
        # Update yaw and pitch
        self.yaw += rotation[0]
        self.pitch = max(min(self.pitch + rotation[1], math.pi / 2), -math.pi / 2)


        #Calculate forward vector
        forward = torch.tensor([
            math.cos(self.pitch) * math.sin(self.yaw),
            math.sin(self.pitch),
            -math.cos(self.pitch) * math.cos(self.yaw)
        ], device='cuda')

        # Calculate movement
        self.position += movement[0] * torch.cross(self.up, forward, dim=0)  # Right vector
        self.position += movement[1] * forward                              # Forward vector
        self.position += movement[2] * self.up                             # Up vector
        #print(self.position, movement)

        self.lookat = self.position + forward
