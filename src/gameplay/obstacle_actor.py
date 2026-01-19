from src.core.actor import Actor
import numpy as np

class ObstacleActor(Actor):
    """
    An actor that represents a spherical obstacle.
    """
    def __init__(self, name="Obstacle", x=0.0, y=0.0, z=0.0, radius=0.005):
        super().__init__(name=name, x=x, y=y, z=z)
        self.radius = radius

    def update(self, dt):
        # Obstacles are static, no update logic needed yet
        pass
