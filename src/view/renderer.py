import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
from lib.gnc import Rzyx

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

    def draw(self, actor):
        # Extract specific state from SubmarineActor if available, else default
        rudder_angle = 0
        stern_angle = 0
        if hasattr(actor, 'rudder_angle'):
            rudder_angle = actor.rudder_angle
        if hasattr(actor, 'stern_angle'):
            stern_angle = actor.stern_angle

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

class TargetPointVisual:
    """Class responsible for drawing the 3D target point."""
    def __init__(self):
        self.quadric = gluNewQuadric()
        gluQuadricDrawStyle(self.quadric, GLU_LINE) # Wireframe

    def draw(self, actor):
        radius = 10.0
        if hasattr(actor, 'radius'):
            radius = actor.radius
        
        glColor3f(0.0, 1.0, 0.0) # Green
        gluSphere(self.quadric, radius, 16, 16)

class UI:
    """Class responsible for rendering the User Interface (HUD)."""
    def __init__(self, font_name="Arial", font_size=18):
        pygame.font.init()
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

    def draw_hud(self, width, height, actor):
        # Only draw HUD for SubmarineActor for now
        if not hasattr(actor, 'get_physics_state'):
            return
            
        physics_state = actor.get_physics_state()
        
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
            f"Pos(XYZ): {eta[0]:.1f}, {eta[1]:.1f}, {eta[2]:.1f}",
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
            self.draw_text_gl(10, 200 + i * 20, text)
            
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def draw_score(self, width, height, episode_reward, total_episodes, success_count, episode_steps):
        """Draw score and episode statistics in the top-left corner"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        success_rate = (success_count / total_episodes * 100) if total_episodes > 0 else 0.0
        
        infos = [
            f"Episode Reward: {episode_reward:.2f}",
            f"Episode Steps: {episode_steps}",
            f"Total Episodes: {total_episodes}",
            f"Success Count: {success_count}",
            f"Success Rate: {success_rate:.1f}%"
        ]
        
        for i, text in enumerate(infos):
            self.draw_text_gl(10, 10 + i * 20, text)
            
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def draw_collision_warning(self, width, height, collision_predicted):
        """Draw collision warning in the top-right corner"""
        if not collision_predicted:
            return
            
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Draw warning text in red
        surface = self.font.render("⚠ COLLISION PREDICTED ⚠", True, (255, 50, 50))
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
        glTexCoord2f(0, 0); glVertex2f(width - w - 10, 10)
        glTexCoord2f(1, 0); glVertex2f(width - 10, 10)
        glTexCoord2f(1, 1); glVertex2f(width - 10, 10 + h)
        glTexCoord2f(0, 1); glVertex2f(width - w - 10, 10 + h)
        glEnd()
        
        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glDeleteTextures([tex_id])
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


class TunnelVisual:
    """Class responsible for drawing the tunnel and obstacles."""
    def __init__(self):
        self.quadric = gluNewQuadric()
        gluQuadricDrawStyle(self.quadric, GLU_LINE)
        self.obstacle_quadric = gluNewQuadric()
    
    def draw(self, actor):
        """Draw the tunnel based on TunnelActor configuration."""
        if not actor.config:
            return
            
        config = actor.config
        center_y = config.center_y
        center_z = config.center_z
        radius = config.radius
        length = config.length
        start_x = config.start_x
        
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.5, 0.8)  # Blue wireframe
        
        # Draw rings along the tunnel
        # Draw every 10 meters
        num_rings = int(length / 10) + 1
        for i in range(num_rings):
            x = start_x + i * 10
            glPushMatrix()
            glTranslate(x, center_y, center_z)
            glRotate(90, 0, 1, 0)  # Align with X axis
            gluDisk(self.quadric, radius - 0.5, radius, 32, 1)
            glPopMatrix()
        
        glEnable(GL_LIGHTING)


class ObstacleVisual:
    """Class responsible for drawing spherical obstacles."""
    def __init__(self):
        self.quadric = gluNewQuadric()
        
    def draw(self, actor):
        # Red sphere
        glColor3f(0.8, 0.2, 0.2)
        gluSphere(self.quadric, actor.radius, 16, 16)


class PointCloudVisual:
    """Class responsible for drawing the LiDAR point cloud."""
    def __init__(self):
        self.point_size = 2.0

    def draw(self, points):
        """
        Draw point cloud.
        Args:
            points: np.ndarray of shape (N, 3) in Body Frame.
        """
        if points is None or len(points) == 0:
            return

        glDisable(GL_LIGHTING)
        glColor3f(0.0, 1.0, 1.0) # Cyan points
        glPointSize(self.point_size)
        
        glBegin(GL_POINTS)
        for p in points:
            glVertex3f(p[0], p[1], p[2])
        glEnd()
        
        glEnable(GL_LIGHTING)


class Renderer:
    def __init__(self, width, height, grid_mode=False, grid_rows=1, grid_cols=1):
        self.width = width
        self.height = height
        self.grid_mode = grid_mode
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        OpenGLUtils.init_gl(width, height)
        self.sub_visual = SubmarineVisual()
        self.target_visual = TargetPointVisual()
        self.obstacle_visual = ObstacleVisual()
        self.tunnel_visual = TunnelVisual()
        self.point_cloud_visual = PointCloudVisual()
        self.ui = UI()

        # Camera State (Free Camera) - per viewport in grid mode
        self.cam_pos = np.array([-10.0, 0.0, 0.0], dtype=np.float32)
        self.cam_yaw = 0.0   # Radians
        self.cam_pitch = 0.0 # Radians
        self.cam_speed = 0.5

        # Grid mode: store camera states for each viewport
        if self.grid_mode:
            self.camera_states = []
            for i in range(grid_rows * grid_cols):
                self.camera_states.append({
                    'cam_pos': np.array([-10.0, 0.0, 0.0], dtype=np.float32),
                    'cam_yaw': 0.0,
                    'cam_pitch': 0.0,
                })

        # Tunnel reference (set by eval_sim.py)
        self.tunnel = None
        self.tunnel_config = None

        # Initial Mouse delta
        pygame.mouse.get_rel()

    def resize(self, width, height):
        self.width = width
        self.height = height
        OpenGLUtils.resize(width, height)

    def _set_viewport(self, row, col):
        """Set viewport for grid cell at (row, col)"""
        cell_width = self.width // self.grid_cols
        cell_height = self.height // self.grid_rows

        # OpenGL viewport origin is bottom-left
        x = col * cell_width
        y = (self.grid_rows - 1 - row) * cell_height

        glViewport(x, y, cell_width, cell_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (cell_width / cell_height) if cell_height > 0 else 1, 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)

    def _draw_viewport_border(self, row, col):
        """Draw border around viewport"""
        cell_width = self.width // self.grid_cols
        cell_height = self.height // self.grid_rows
        x = col * cell_width
        y = row * cell_height

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glColor3f(0.3, 0.5, 0.7)
        glLineWidth(2)

        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + cell_width, y)
        glVertex2f(x + cell_width, y + cell_height)
        glVertex2f(x, y + cell_height)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _get_camera_state(self, viewport_idx):
        """Get camera state for specific viewport"""
        if self.grid_mode and viewport_idx < len(self.camera_states):
            return self.camera_states[viewport_idx]
        return {
            'cam_pos': self.cam_pos,
            'cam_yaw': self.cam_yaw,
            'cam_pitch': self.cam_pitch,
        }

    def update_grid_camera_input(self, viewport_idx):
        """Update camera for specific viewport based on mouse position"""
        if not self.grid_mode or viewport_idx >= len(self.camera_states):
            return

        cell_width = self.width // self.grid_cols
        cell_height = self.height // self.grid_rows

        mx, my = pygame.mouse.get_pos()
        col = mx // cell_width
        row = (self.height - 1 - my) // cell_height

        # Only update if mouse is in this viewport
        target_row = viewport_idx // self.grid_cols
        target_col = viewport_idx % self.grid_cols

        if row == target_row and col == target_col:
            cam_state = self.camera_states[viewport_idx]

            # Mouse Rotation (Right Click)
            if pygame.mouse.get_pressed()[2]:
                mx_rel, my_rel = pygame.mouse.get_rel()
                cam_state['cam_yaw'] += mx_rel * 0.002
                cam_state['cam_pitch'] += my_rel * 0.002
                cam_state['cam_pitch'] = max(min(cam_state['cam_pitch'], 1.5), -1.5)
            else:
                pygame.mouse.get_rel()

            # Keyboard Movement (WASD) - shared across all viewports
            keys = pygame.key.get_pressed()

            speed = cam_state.get('speed', 0.5)
            if keys[K_LSHIFT] or keys[K_RSHIFT]:
                speed *= 2.0

            if keys[K_w]:
                cam_state['cam_pos'][0] += math.cos(cam_state['cam_yaw']) * math.cos(cam_state['cam_pitch']) * speed
                cam_state['cam_pos'][1] += math.sin(cam_state['cam_yaw']) * math.cos(cam_state['cam_pitch']) * speed
                cam_state['cam_pos'][2] += -math.sin(cam_state['cam_pitch']) * speed

            if keys[K_s]:
                cam_state['cam_pos'][0] -= math.cos(cam_state['cam_yaw']) * math.cos(cam_state['cam_pitch']) * speed
                cam_state['cam_pos'][1] -= math.sin(cam_state['cam_yaw']) * math.cos(cam_state['cam_pitch']) * speed
                cam_state['cam_pos'][2] -= -math.sin(cam_state['cam_pitch']) * speed

            if keys[K_a]:
                cam_state['cam_pos'][0] += math.sin(cam_state['cam_yaw']) * speed
                cam_state['cam_pos'][1] += -math.cos(cam_state['cam_yaw']) * speed

            if keys[K_d]:
                cam_state['cam_pos'][0] -= math.sin(cam_state['cam_yaw']) * speed
                cam_state['cam_pos'][1] -= -math.cos(cam_state['cam_yaw']) * speed

            if keys[K_e]:
                cam_state['cam_pos'][2] += speed
            if keys[K_q]:
                cam_state['cam_pos'][2] -= speed

    def render_grid(self, logic_stages_list, point_clouds_list, score_data_list):
        """Render multiple environments in grid layout"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for idx, (logic_stage, point_cloud, score_data) in enumerate(zip(logic_stages_list, point_clouds_list, score_data_list)):
            row = idx // self.grid_cols
            col = idx % self.grid_cols

            self._set_viewport(row, col)

            # Update camera for this viewport
            if self.grid_mode:
                cam_state = self.camera_states[idx]
                cx, cy, cz = cam_state['cam_pos']
                cam_yaw = cam_state['cam_yaw']
                cam_pitch = cam_state['cam_pitch']
            else:
                cx, cy, cz = self.cam_pos
                cam_yaw = self.cam_yaw
                cam_pitch = self.cam_pitch

            glLoadIdentity()

            lx = cx + math.cos(cam_yaw) * math.cos(cam_pitch)
            ly = cy + math.sin(cam_yaw) * math.cos(cam_pitch)
            lz = cz - math.sin(cam_pitch)

            gluLookAt(cx, cy, cz, lx, ly, lz, 0, 0, -1)
            glLightfv(GL_LIGHT0, GL_POSITION, (cx, cy, cz, 1))

            # Draw scene for this viewport
            if self.tunnel is None:
                glPushMatrix()
                grid_step = 20
                grid_x = round(cx / grid_step) * grid_step
                grid_y = round(cy / grid_step) * grid_step
                glTranslate(grid_x, grid_y, 0)
                OpenGLUtils.draw_grid(size=100, step=5)
                glPopMatrix()

            actors = logic_stage.get_actors()
            main_actor = actors[0] if actors else None

            for actor in actors:
                glPushMatrix()
                glTranslate(actor.position[0], actor.position[1], actor.position[2])
                glRotate(math.degrees(actor.orientation[2]), 0, 0, 1)
                glRotate(math.degrees(actor.orientation[1]), 0, 1, 0)
                glRotate(math.degrees(actor.orientation[0]), 1, 0, 0)

                actor_type = type(actor).__name__
                if actor_type == 'TargetPointActor':
                    self.target_visual.draw(actor)
                elif actor_type == 'ObstacleActor':
                    self.obstacle_visual.draw(actor)
                elif actor_type == 'TunnelActor':
                    self.tunnel_visual.draw(actor)
                else:
                    self.sub_visual.draw(actor)
                    if actor == main_actor and point_cloud is not None:
                        self.point_cloud_visual.draw(point_cloud)

                glPopMatrix()

        # Draw borders after all viewports
        for idx in range(len(logic_stages_list)):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            self._set_viewport(row, col)
            self._draw_viewport_border(row, col)

        # Draw UI overlays (full viewport for each)
        cell_width = self.width // self.grid_cols
        cell_height = self.height // self.grid_rows

        for idx, score_data in enumerate(score_data_list):
            row = idx // self.grid_cols
            col = idx % self.grid_cols

            # Draw viewport label and stats
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, cell_width, cell_height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            glDisable(GL_LIGHTING)
            glDisable(GL_DEPTH_TEST)

            # Environment label
            label_text = f"Env {idx}"
            surface = self.ui.font.render(label_text, True, (255, 255, 0))
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
            glColor3f(1, 1, 1)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(5, 5)
            glTexCoord2f(1, 0); glVertex2f(5 + w, 5)
            glTexCoord2f(1, 1); glVertex2f(5 + w, 5 + h)
            glTexCoord2f(0, 1); glVertex2f(5, 5 + h)
            glEnd()
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
            glDeleteTextures([tex_id])

            # Stats for this environment
            if score_data:
                stats = [
                    f"Reward: {score_data.get('episode_reward', 0):.1f}",
                    f"Steps: {score_data.get('episode_steps', 0)}",
                    f"Status: {'OK' if not score_data.get('collision_predicted', False) else 'COLLISION!'}",
                ]
                for i, stat in enumerate(stats):
                    surface = self.ui.font.render(stat, True, (200, 200, 200))
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
                    glColor3f(1, 1, 1)
                    glBegin(GL_QUADS)
                    glTexCoord2f(0, 0); glVertex2f(5, 25 + i * 18)
                    glTexCoord2f(1, 0); glVertex2f(5 + w, 25 + i * 18)
                    glTexCoord2f(1, 1); glVertex2f(5 + w, 25 + i * 18 + h)
                    glTexCoord2f(0, 1); glVertex2f(5, 25 + i * 18 + h)
                    glEnd()
                    glDisable(GL_BLEND)
                    glDisable(GL_TEXTURE_2D)
                    glDeleteTextures([tex_id])

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)

        pygame.display.flip()

    def update_camera_input(self):
        # Mouse Rotation (Right Click)
        if pygame.mouse.get_pressed()[2]: # Right click
            mx, my = pygame.mouse.get_rel()
            self.cam_yaw += mx * 0.002
            self.cam_pitch += my * 0.002
            self.cam_pitch = max(min(self.cam_pitch, 1.5), -1.5)
        else:
            pygame.mouse.get_rel()
            
        # Keyboard Movement (WASD)
        keys = pygame.key.get_pressed()
        
        # Calculate forward and right vectors based on yaw
        # Forward vector in horizontal plane
        fwd_x = math.cos(self.cam_yaw)
        fwd_y = math.sin(self.cam_yaw)
        # fwd_z = 0
        
        # Right vector
        rgt_x = math.sin(self.cam_yaw)
        rgt_y = -math.cos(self.cam_yaw)
        
        speed = self.cam_speed
        if keys[K_LSHIFT] or keys[K_RSHIFT]:
            speed *= 2.0
            
        if keys[K_w]:
            self.cam_pos[0] += fwd_x * speed
            self.cam_pos[1] += fwd_y * speed
            self.cam_pos[2] -= math.sin(self.cam_pitch) * speed # Move in look direction including pitch? Or just horizontal?
            # Usually FPS camera moves horizontally on W/S, but "free cam" might fly.
            # Let's make it fly in the look direction.
            # Re-calculating proper 3D forward vector
            # fx = cos(yaw)cos(pitch), fy = sin(yaw)cos(pitch), fz = -sin(pitch)
            self.cam_pos[0] += math.cos(self.cam_yaw) * math.cos(self.cam_pitch) * speed
            self.cam_pos[1] += math.sin(self.cam_yaw) * math.cos(self.cam_pitch) * speed
            self.cam_pos[2] += -math.sin(self.cam_pitch) * speed

        if keys[K_s]:
            self.cam_pos[0] -= math.cos(self.cam_yaw) * math.cos(self.cam_pitch) * speed
            self.cam_pos[1] -= math.sin(self.cam_yaw) * math.cos(self.cam_pitch) * speed
            self.cam_pos[2] -= -math.sin(self.cam_pitch) * speed

        if keys[K_a]:
            # Strafe Left
            # Right vector is (sin(yaw), -cos(yaw), 0)
            # Left is (-sin(yaw), cos(yaw), 0)
            self.cam_pos[0] += math.sin(self.cam_yaw) * speed
            self.cam_pos[1] += -math.cos(self.cam_yaw) * speed

        if keys[K_d]:
            # Strafe Right
            self.cam_pos[0] -= math.sin(self.cam_yaw) * speed
            self.cam_pos[1] -= -math.cos(self.cam_yaw) * speed
            
        # Up/Down (E/Q or Space/Ctrl) - Optional
        if keys[K_e]:
            self.cam_pos[2] += speed
        if keys[K_q]:
            self.cam_pos[2] -= speed

    def render(self, logic_stage, point_cloud=None, score_data=None):
        self.update_camera_input()
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera LookAt
        # Eye position: self.cam_pos
        # Target position: self.cam_pos + forward_vector
        
        cx = self.cam_pos[0]
        cy = self.cam_pos[1]
        cz = self.cam_pos[2]
        
        lx = cx + math.cos(self.cam_yaw) * math.cos(self.cam_pitch)
        ly = cy + math.sin(self.cam_yaw) * math.cos(self.cam_pitch)
        lz = cz - math.sin(self.cam_pitch)
        
        gluLookAt(cx, cy, cz,
                  lx, ly, lz,
                  0, 0, -1) # Z is up? No, in this coord system Z seems to be...
                  # In OpenGLUtils.draw_axes: Z is Blue.
                  # In SubmarineVisual: Cylinder length 1.6 along Z (then rotated).
                  # In SubmarineEnv: spawn_pos [x, 0, 0], eta[2] is depth?
                  # Usually NED: x North, y East, z Down.
                  # OpenGL default: y Up, -z Forward.
                  # Let's check `draw_axes`: X Red, Y Green, Z Blue.
                  # Submarine hull: Cylinder along Z, rotated 90 around Y -> Along X.
                  # So X is forward for submarine.
                  # Fin logic: Top fin +Y. Starboard +Z.
                  # This suggests Y is Up/Down (Top fin), Z is Right/Left (Starboard).
                  # If Y is Up, then gluLookAt up vector should be (0, 1, 0) or (0, 0, -1) if Z is depth (Down).
                  # Let's check previous code: gluLookAt(..., 0, 0, -1).
                  # So Up vector is (0, 0, -1). This implies Z is Down (Depth).
                  # And Y is... wait.
                  # Previous code:
                  # cx_b = dist * cos(pitch) * cos(yaw)
                  # cy_b = dist * cos(pitch) * sin(yaw)
                  # cz_b = -dist * sin(pitch)
                  # If pitch=0, z=0. If pitch positive (look up?), z negative (up?).
                  # Let's stick to (0, 0, -1) as UP if the world is NED (Z down).
        
        # Light
        glLightfv(GL_LIGHT0, GL_POSITION, (cx, cy, cz, 1)) # Light at camera
        
        # Draw tunnel if available
        # ... (same as before)
        
        # Grid
        if self.tunnel is None:
            glPushMatrix()
            grid_step = 20
            grid_x = round(cx / grid_step) * grid_step
            grid_y = round(cy / grid_step) * grid_step
            glTranslate(grid_x, grid_y, 0)
            OpenGLUtils.draw_grid(size=100, step=5)
            glPopMatrix()
        
        # Draw all actors
        actors = logic_stage.get_actors()
        main_actor = actors[0] if actors else None
        
        for actor in actors:
            glPushMatrix()
            glTranslate(actor.position[0], actor.position[1], actor.position[2])
            glRotate(math.degrees(actor.orientation[2]), 0, 0, 1) # Yaw
            glRotate(math.degrees(actor.orientation[1]), 0, 1, 0) # Pitch
            glRotate(math.degrees(actor.orientation[0]), 1, 0, 0) # Roll
            
            # Select visual based on actor type/name
            actor_type = type(actor).__name__
            if actor_type == 'TargetPointActor':
                self.target_visual.draw(actor)
            elif actor_type == 'ObstacleActor':
                self.obstacle_visual.draw(actor)
            elif actor_type == 'TunnelActor':
                self.tunnel_visual.draw(actor)
            else:
                self.sub_visual.draw(actor)
                # Draw point cloud attached to submarine
                if actor == main_actor and point_cloud is not None:
                    self.point_cloud_visual.draw(point_cloud)
            
            # OpenGLUtils.draw_axes(2.0)
            glPopMatrix()
            
        # Draw score (top-left)
        if score_data is not None:
            self.ui.draw_score(
                self.width, self.height,
                score_data['episode_reward'],
                score_data['total_episodes'],
                score_data['success_count'],
                score_data['episode_steps']
            )
            
            # Draw collision warning (top-right)
            if 'collision_predicted' in score_data:
                self.ui.draw_collision_warning(
                    self.width, self.height,
                    score_data['collision_predicted']
                )
            
        # Draw HUD for main actor (below score)
        if main_actor:
            self.ui.draw_hud(self.width, self.height, main_actor)
        
        pygame.display.flip()

