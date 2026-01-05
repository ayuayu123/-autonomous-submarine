import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import sys
from torpedo import torpedo
from lib.gnc import attitudeEuler, Rzyx

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)

class OpenGLUtils:
    """Helper class for OpenGL initialization and basic drawing."""
    @staticmethod
    def init_gl(width, height):
        glClearColor(0.0, 0.05, 0.1, 1.0) # Deep ocean blue background
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Light configuration
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        OpenGLUtils.resize(width, height)

    @staticmethod
    def resize(width, height):
        if height == 0: height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (width / height), 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)

    @staticmethod
    def draw_axes(length=1.0):
        glBegin(GL_LINES)
        # X - Red
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)
        # Y - Green
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, length, 0)
        # Z - Blue
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        glEnd()

    @staticmethod
    def draw_grid(size=20, step=2):
        glDisable(GL_LIGHTING)
        glColor3f(0.0, 0.2, 0.4)
        glBegin(GL_LINES)
        for i in range(-size, size+1, step):
            glVertex3f(i, -size, 0)
            glVertex3f(i, size, 0)
            glVertex3f(-size, i, 0)
            glVertex3f(size, i, 0)
        glEnd()
        glEnable(GL_LIGHTING)

class SubmarineVisual:
    """Class responsible for drawing the 3D submarine model."""
    def __init__(self):
        self.quadric = gluNewQuadric()

    def draw(self, rudder_angle, stern_angle):
        # --- Hull (Cylinder) ---
        glColor3f(0.7, 0.7, 0.75) # Light grey hull
        glPushMatrix()
        glTranslate(-0.8, 0, 0) # Center the 1.6m length
        # Cylinder draws along Z by default. Rotate to align with X.
        glRotate(90, 0, 1, 0) 
        gluCylinder(self.quadric, 0.19/2, 0.19/2, 1.6, 32, 16)
        
        # --- Nose (Cone) ---
        glTranslate(0, 0, 1.6) # Move to front of cylinder
        glColor3f(0.8, 0.3, 0.3) # Red nose
        gluCylinder(self.quadric, 0.19/2, 0.0, 0.3, 32, 4)
        glPopMatrix()
        
        # --- Tail Cap (Disk) ---
        glPushMatrix()
        glTranslate(-0.8, 0, 0)
        glRotate(-90, 0, 1, 0) # Rotate to face back
        glColor3f(0.6, 0.6, 0.6)
        gluDisk(self.quadric, 0, 0.19/2, 32, 1)
        glPopMatrix()
        
        # --- Fins ---
        # Top Fin (Up in GL is +Y). Deflects for Rudder.
        self._draw_fin(0, -rudder_angle) 
        # Bottom Fin (-Y). 
        self._draw_fin(180, -rudder_angle)
        # Starboard Fin (+Z). Angle = 90.
        self._draw_fin(90, stern_angle)
        # Port Fin (-Z). Angle = -90.
        self._draw_fin(-90, stern_angle)

    def _draw_fin(self, angle_deg, deflect_angle):
        fin_pos_x = -0.8
        fin_w = 0.2
        fin_h = 0.3
        
        glPushMatrix()
        glTranslate(fin_pos_x, 0, 0)
        glRotate(angle_deg, 1, 0, 0) # Rotate around X
        glTranslate(0, 0.19/2, 0) # Move to surface
        
        # Deflection
        glRotate(math.degrees(deflect_angle), 0, 1, 0) # Rotate around span (Y)
        
        glColor3f(1.0, 0.8, 0.0) # Yellow fins
        glBegin(GL_TRIANGLES)
        glVertex3f(0, 0, 0) # Base Front
        glVertex3f(-fin_w, 0, 0) # Base Back
        glVertex3f(-fin_w/2, fin_h, 0) # Tip
        glEnd()
        
        glPopMatrix()

class UI:
    """Class responsible for rendering the User Interface (HUD)."""
    def __init__(self, font_name="Arial", font_size=18):
        self.font = pygame.font.SysFont(font_name, font_size)

    def draw_text_gl(self, x, y, text):
        surface = self.font.render(text, True, (255, 255, 255))
        text_data = pygame.image.tostring(surface, "RGBA", False)
        w, h = surface.get_size()
        
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_LIGHTING)
        
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 1); glVertex2f(x, y + h)
        glEnd()
        
        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glDeleteTextures([tex_id])

    def draw_hud(self, width, height, physics_state):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        u_actual = physics_state['u_actual']
        eta = physics_state['eta']
        nu = physics_state['nu']
        target_rpm = physics_state['target_rpm']
        
        top_deg = math.degrees(u_actual[0])
        bot_deg = math.degrees(u_actual[1])
        stb_deg = math.degrees(u_actual[2])
        prt_deg = math.degrees(u_actual[3])
        
        infos = [
            f"RPM: {target_rpm:.1f} / {u_actual[4]:.1f}",
            f"Depth: {eta[2]:.2f} m",
            f"Heading: {math.degrees(eta[5]):.1f} deg",
            f"Pitch: {math.degrees(eta[4]):.1f} deg",
            f"Speed: {np.linalg.norm(nu[0:3]):.2f} m/s",
            f"Rudders: {top_deg:.1f} / {bot_deg:.1f}",
            f"Sterns: {stb_deg:.1f} / {prt_deg:.1f}",
            "WASD: Steer | Arrows: RPM | Mouse R-Click: Cam"
        ]
        
        for i, text in enumerate(infos):
            self.draw_text_gl(10, 10 + i * 20, text)
            
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

class PhysicsInteraction:
    """Class responsible for physics simulation and control input processing."""
    def __init__(self):
        # Init vehicle
        self.vehicle = torpedo(controlSystem="stepInput", r_rpm=0)
        
        # State
        self.eta = np.zeros(6) # [x, y, z, phi, theta, psi]
        self.nu = np.zeros(6)
        self.u_actual = np.zeros(self.vehicle.dimU)
        self.u_control = np.zeros(self.vehicle.dimU)
        
        self.eta[2] = 5.0 # Initial depth
        
        # Control State
        self.target_rpm = 0.0
        self.rudder_angle = 0.0
        self.stern_angle = 0.0
        self.fin_offsets = np.zeros(4)
        
        # Constants
        self.max_angle = 20 * math.pi / 180
        self.angle_step = 0.5 * math.pi / 180
        self.fine_step = 0.1 * math.pi / 180

    def process_input(self, keys):
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

    def get_state(self):
        return {
            'eta': self.eta,
            'nu': self.nu,
            'u_actual': self.u_actual,
            'target_rpm': self.target_rpm,
            'rudder_angle': self.rudder_angle,
            'stern_angle': self.stern_angle
        }

class SimulatorApp:
    """Main Application class integrating UI, Visuals and Physics."""
    def __init__(self):
        pygame.init()
        self.width, self.height = 800, 600
        
        # Set up OpenGL context
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("3D Submarine Simulator (OpenGL)")
        
        OpenGLUtils.init_gl(self.width, self.height)
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Subsystems
        self.ui = UI()
        self.sub_visual = SubmarineVisual()
        self.physics = PhysicsInteraction()
        
        # Camera State
        self.cam_dist = 6.0
        self.cam_yaw_rel = math.pi 
        self.cam_pitch_rel = -0.3
        
        pygame.mouse.get_rel() # Init mouse delta

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                OpenGLUtils.resize(self.width, self.height)

    def update_camera(self):
        if pygame.mouse.get_pressed()[2]: # Right click
            mx, my = pygame.mouse.get_rel()
            self.cam_yaw_rel += mx * 0.005
            self.cam_pitch_rel += my * 0.005
            self.cam_pitch_rel = max(min(self.cam_pitch_rel, 1.5), -1.5)
        else:
            pygame.mouse.get_rel()

    def run(self):
        while self.running:
            dt_ms = self.clock.tick(60)
            dt = dt_ms / 1000.0
            if dt > 0.1: dt = 0.1
            
            self.handle_events()
            
            # Input & Physics
            keys = pygame.key.get_pressed()
            self.physics.process_input(keys)
            self.physics.update(dt)
            self.update_camera()
            
            # Rendering
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            state = self.physics.get_state()
            eta = state['eta']
            
            # Camera Position Calculation
            cx_b = self.cam_dist * math.cos(self.cam_pitch_rel) * math.cos(self.cam_yaw_rel)
            cy_b = self.cam_dist * math.cos(self.cam_pitch_rel) * math.sin(self.cam_yaw_rel)
            cz_b = -self.cam_dist * math.sin(self.cam_pitch_rel)
            cam_offset_body = np.array([cx_b, cy_b, cz_b])
            
            R_vehicle = Rzyx(eta[3], eta[4], eta[5])
            cam_pos_world = eta[0:3] + R_vehicle @ cam_offset_body
            
            gluLookAt(cam_pos_world[0], cam_pos_world[1], cam_pos_world[2],
                      eta[0], eta[1], eta[2],
                      0, 0, -1)
            
            # Light
            glLightfv(GL_LIGHT0, GL_POSITION, (eta[0]+10, eta[1]+10, eta[2]-20, 1))
            
            # Grid
            glPushMatrix()
            grid_step = 20
            grid_x = round(eta[0] / grid_step) * grid_step
            grid_y = round(eta[1] / grid_step) * grid_step
            glTranslate(grid_x, grid_y, 0)
            OpenGLUtils.draw_grid(size=100, step=5)
            glPopMatrix()
            
            # Submarine
            glPushMatrix()
            glTranslate(eta[0], eta[1], eta[2])
            glRotate(math.degrees(eta[5]), 0, 0, 1) # Yaw
            glRotate(math.degrees(eta[4]), 0, 1, 0) # Pitch
            glRotate(math.degrees(eta[3]), 1, 0, 0) # Roll
            
            self.sub_visual.draw(state['rudder_angle'], state['stern_angle'])
            OpenGLUtils.draw_axes(2.0)
            glPopMatrix()
            
            # HUD
            self.ui.draw_hud(self.width, self.height, state)
            
            pygame.display.flip()
            
        pygame.quit()

if __name__ == "__main__":
    app = SimulatorApp()
    app.run()
