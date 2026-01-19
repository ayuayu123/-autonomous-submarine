from src.core.actor import Actor
import numpy as np

class TargetPointActor(Actor):
    """
    An actor that detects when a target actor enters its range.
    """
    def __init__(self, name="TargetPoint", x=0.0, y=0.0, z=0.0, radius=10.0, target_actor=None):
        super().__init__(name=name, x=x, y=y, z=z)
        self.radius = radius
        self.target_actor = target_actor
        self.is_inside = False
        
    def update(self, dt):
        if not self.target_actor:
            return
            
        # Calculate distance to target
        dist = np.linalg.norm(self.position - self.target_actor.position)
        
        if dist <= self.radius:
            if not self.is_inside:
                self.on_enter()
                self.is_inside = True
        else:
            self.is_inside = False
            
    def on_enter(self):
        print(f"[{self.name}] TRIGGERED! Target entered radius {self.radius}. Distance: {np.linalg.norm(self.position - self.target_actor.position):.2f}")
