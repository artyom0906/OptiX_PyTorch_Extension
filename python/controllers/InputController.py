from abc import ABC, abstractmethod

class InputController(ABC):
    @abstractmethod
    def update(self):
        """Update input states."""
        pass

    @abstractmethod
    def get_movement(self):
        """Get movement input as (x, y, z)."""
        pass

    @abstractmethod
    def get_rotation(self):
        """Get rotation input as (yaw, pitch)."""
        pass
