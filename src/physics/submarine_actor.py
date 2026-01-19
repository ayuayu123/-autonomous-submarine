import numpy as np
import math
import sys
import os

# Add project root to path to find torpedo and lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.actor import Actor
from torpedo import torpedo
from lib.gnc import attitudeEuler

import pygame

class SubmarineActor(Actor):
    """
    Actor representing the submarine with physics simulation.
    """
    def __init__(self, name="Submarine"):
        super().__init__(name=name)
        
        # Init vehicle
        self.vehicle = torpedo(controlSystem="stepInput", r_rpm=0)
        
        # State (eta: [x, y, z, phi, theta, psi])
        self.eta = np.zeros(6) 
        # Initial position (0, 0, 400) - 1 unit = 1 meter
        self.eta[0] = 0.0
        self.eta[1] = 0.0
        self.eta[2] = 100.0
        
        self.nu = np.zeros(6)
        self.u_actual = np.zeros(self.vehicle.dimU)
        self.u_control = np.zeros(self.vehicle.dimU)
        
        # Control State
        self.target_rpm = 0.0
        self.rudder_angle = 0.0
        self.stern_angle = 0.0
        self.fin_offsets = np.zeros(4)
        
        # Constants
        self.max_angle = 20 * math.pi / 180
        self.angle_step = 0.5 * math.pi / 180
        self.fine_step = 0.1 * math.pi / 180
        
        # Sync base Actor state
        self._sync_state()

    def _sync_state(self):
        """Sync internal physics state to base Actor properties."""
        self.position = self.eta[0:3]
        self.orientation = self.eta[3:6] # roll, pitch, yaw

    def process_input(self, keys):
        """
        Process keyboard input to control the submarine.
        keys: pygame.key.get_pressed() result
        """
        # RPM
        if keys[pygame.K_UP]: self.target_rpm += 10
        if keys[pygame.K_DOWN]: self.target_rpm -= 10
        self.target_rpm = max(min(self.target_rpm, 1525), -1525)
        
        # Steering
        if keys[pygame.K_w]: self.stern_angle += self.angle_step
        if keys[pygame.K_s]: self.stern_angle -= self.angle_step
        if keys[pygame.K_a]: self.rudder_angle += self.angle_step
        if keys[pygame.K_d]: self.rudder_angle -= self.angle_step
        
        # Fine tune
        direction = -1 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1
        if keys[pygame.K_1]: self.fin_offsets[0] += direction * self.fine_step
        if keys[pygame.K_2]: self.fin_offsets[1] += direction * self.fine_step
        if keys[pygame.K_3]: self.fin_offsets[2] += direction * self.fine_step
        if keys[pygame.K_4]: self.fin_offsets[3] += direction * self.fine_step
        
        # Clamping
        self.stern_angle = max(min(self.stern_angle, self.max_angle), -self.max_angle)
        self.rudder_angle = max(min(self.rudder_angle, self.max_angle), -self.max_angle)

    def update(self, dt):
        # Map controls
        self.u_control[0] = -self.rudder_angle + self.fin_offsets[0]
        self.u_control[1] = self.rudder_angle + self.fin_offsets[1]
        self.u_control[2] = -self.stern_angle + self.fin_offsets[2]
        self.u_control[3] = self.stern_angle + self.fin_offsets[3]
        self.u_control[4] = self.target_rpm
        
        # Dynamics
        self.nu, self.u_actual = self.vehicle.dynamics(self.eta, self.nu, self.u_actual, self.u_control, dt)
        self.eta = attitudeEuler(self.eta, self.nu, dt)
        
        # Sync to base class
        self._sync_state()

    def get_physics_state(self):
        return {
            'eta': self.eta,
            'nu': self.nu,
            'u_actual': self.u_actual,
            'target_rpm': self.target_rpm,
            'rudder_angle': self.rudder_angle,
            'stern_angle': self.stern_angle
        }
