import numpy as np

class Actor:
    """
    Base class for all entities in the Logic Stagnation Zone (Logic Stage).
    Basic parameters: XYZ (Position) and Euler Angles (Orientation).
    """
    def __init__(self, name="Actor", x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        self.name = name
        # Position: [x, y, z] (North, East, Down in generic context, or standard 3D)
        self.position = np.array([x, y, z], dtype=float)
        # Orientation: [roll, pitch, yaw] (Radians)
        self.orientation = np.array([roll, pitch, yaw], dtype=float)

    def set_position(self, x, y, z):
        self.position = np.array([x, y, z], dtype=float)

    def set_orientation(self, roll, pitch, yaw):
        self.orientation = np.array([roll, pitch, yaw], dtype=float)

    def update(self, dt):
        """
        Update logic for the actor. 
        Override this in subclasses for specific behavior (e.g., Physics).
        """
        pass
