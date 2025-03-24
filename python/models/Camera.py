import math

import numpy as np
import torch


class Camera:
    def __init__(self, position, lookat, up):
        # Keep tensors on CPU - ResourceManager will handle GPU transfer
        self.position = position.clone() if isinstance(position, torch.Tensor) else torch.tensor(position, dtype=torch.float32)
        self.lookat = lookat.clone() if isinstance(lookat, torch.Tensor) else torch.tensor(lookat, dtype=torch.float32)
        self.up = up.clone() if isinstance(up, torch.Tensor) else torch.tensor(up, dtype=torch.float32)
        self.yaw = 0.0
        self.pitch = 0.0

    def update(self, movement, rotation):
        # Update yaw and pitch with better sensitivity control
        self.yaw += rotation[0] * 0.5  # Lower sensitivity for more precise control
        # Clamp pitch to prevent flipping (leaving a little room at the extremes)
        self.pitch = max(min(self.pitch + rotation[1] * 0.5, math.pi / 2 - 0.01), -math.pi / 2 + 0.01)


        #Calculate forward vector
        forward = torch.tensor([
            math.cos(self.pitch) * math.sin(self.yaw),
            math.sin(self.pitch),
            -math.cos(self.pitch) * math.cos(self.yaw)
        ], dtype=torch.float32)

        # Calculate movement
        self.position += movement[0] * torch.cross(self.up, forward, dim=0)  # Right vector
        self.position += movement[1] * forward                              # Forward vector
        self.position += movement[2] * self.up                             # Up vector
        # Uncomment for debugging movement
        # print(f"Camera position: {self.position}")

        self.lookat = self.position + forward
