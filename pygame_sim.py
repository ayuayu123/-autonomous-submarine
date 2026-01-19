import pygame
from pygame.locals import *
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(__file__))

from src.core.stage import LogicStage
from src.physics.submarine_actor import SubmarineActor
from src.gameplay.target_point_actor import TargetPointActor
from src.view.renderer import Renderer

class SimulatorApp:
    """Main Application class integrating Logic, Physics and View."""
    def __init__(self):
        pygame.init()
        self.width, self.height = 1260, 900
        
        # Set up OpenGL context
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("3D Submarine Simulator (Refactored)")
        
        # 1. Logic Layer (Stagnation Zone)
        self.logic_stage = LogicStage()
        
        # 2. Physics Layer (Actor)
        self.submarine = SubmarineActor("HeroSub")
        self.logic_stage.add_actor(self.submarine)
        
        # Add Target Point
        self.target = TargetPointActor("Target1", x=40.0, y=0.0, z=100.0, radius=10.0, target_actor=self.submarine)
        self.logic_stage.add_actor(self.target)
        
        # 3. Animation/View Layer
        self.renderer = Renderer(self.width, self.height)
        
        self.clock = pygame.time.Clock()
        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                self.renderer.resize(self.width, self.height)

    def run(self):
        while self.running:
            dt_ms = self.clock.tick(60)
            dt = dt_ms / 1000.0
            if dt > 0.1: dt = 0.1
            
            self.handle_events()
            
            # Input Processing
            keys = pygame.key.get_pressed()
            # Pass input to the submarine actor
            # In a full system, we might route this through an InputManager in the LogicStage
            self.submarine.process_input(keys)
            
            # Logic Update
            self.logic_stage.update(dt)
            
            # Rendering
            self.renderer.render(self.logic_stage)
            
        pygame.quit()

if __name__ == "__main__":
    app = SimulatorApp()
    app.run()
