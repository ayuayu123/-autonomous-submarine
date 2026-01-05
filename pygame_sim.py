import pygame
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

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pos = np.array([-5.0, 0.0, -2.0]) # Initial camera position (behind and above)
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 0.0, -1.0]) # NED: z is down, so -z is up
        self.fov = 60
        self.aspect = width / height
        self.near = 0.1
        self.far = 100.0

    def get_view_matrix(self):
        # Simple look-at matrix
        z_axis = self.target - self.pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(z_axis, self.up)
        if np.linalg.norm(x_axis) < 1e-6:
             x_axis = np.array([1.0, 0.0, 0.0]) # Fallback
        else:
             x_axis = x_axis / np.linalg.norm(x_axis)
             
        y_axis = np.cross(x_axis, z_axis)
        
        # View matrix (World -> Camera)
        # R = [x_axis, y_axis, z_axis]^T
        R = np.array([x_axis, y_axis, z_axis])
        t = -R @ self.pos
        
        V = np.identity(4)
        V[:3, :3] = R
        V[:3, 3] = t
        return V

    def project(self, points_3d):
        # points_3d: N x 3
        # Transform to camera space
        view = self.get_view_matrix()
        
        # Homogeneous coordinates
        ones = np.ones((len(points_3d), 1))
        points_4d = np.hstack([points_3d, ones])
        
        # Camera space
        cam_points = points_4d @ view.T
        
        # Project to screen
        # We look down +Z in camera space? 
        # Wait, my get_view_matrix constructs Z pointing TO target.
        # Usually OpenGL Camera looks down -Z. 
        # Let's adjust: Z should be forward.
        # z_axis = target - pos (Forward)
        
        # Perspective projection
        # x' = x / z * f
        # y' = y / z * f
        
        # Simple projection
        f = self.width / (2 * math.tan(math.radians(self.fov) / 2))
        
        projected = []
        for p in cam_points:
            x, y, z, w = p
            # Check if point is behind camera
            if z <= self.near: 
                projected.append(None)
                continue
                
            px = x * f / z + self.width / 2
            py = -y * f / z + self.height / 2 # Flip Y for screen coords
            projected.append((px, py))
            
        return projected

def create_submarine_mesh():
    # Simple cylinder + cone
    vertices = []
    edges = []
    
    # Body (Cylinder)
    segments = 8
    radius = 0.2
    length = 1.6
    x_front = length / 2
    x_back = -length / 2
    
    # Front circle
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        y = radius * math.cos(angle)
        z = radius * math.sin(angle)
        vertices.append([x_front, y, z])
        
    # Back circle
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        y = radius * math.cos(angle)
        z = radius * math.sin(angle)
        vertices.append([x_back, y, z])
        
    # Nose (Cone)
    vertices.append([x_front + 0.3, 0, 0])
    nose_idx = len(vertices) - 1
    
    # Connect edges
    for i in range(segments):
        # Front circle
        edges.append((i, (i + 1) % segments))
        # Back circle
        edges.append((segments + i, segments + (i + 1) % segments))
        # Connecting lines
        edges.append((i, segments + i))
        # Nose
        edges.append((i, nose_idx))
        
    # Fins (Simple triangles)
    # Top Rudder
    vertices.append([x_back, 0, -radius]) # Base
    vertices.append([x_back - 0.2, 0, -radius]) # Base back
    vertices.append([x_back - 0.1, 0, -radius - 0.3]) # Tip
    
    # Bottom Rudder
    vertices.append([x_back, 0, radius])
    vertices.append([x_back - 0.2, 0, radius])
    vertices.append([x_back - 0.1, 0, radius + 0.3])
    
    # Right Stern (Starboard) - Y is right?
    # NED: x North, y East (Right), z Down.
    vertices.append([x_back, radius, 0])
    vertices.append([x_back - 0.2, radius, 0])
    vertices.append([x_back - 0.1, radius + 0.3, 0])
    
    # Left Stern (Port)
    vertices.append([x_back, -radius, 0])
    vertices.append([x_back - 0.2, -radius, 0])
    vertices.append([x_back - 0.1, -radius - 0.3, 0])
    
    # Add fin edges manually or just draw all vertices as lines?
    # Let's just draw all defined edges plus fin edges
    base_idx = nose_idx + 1
    # Top fin
    edges.append((base_idx, base_idx+1))
    edges.append((base_idx+1, base_idx+2))
    edges.append((base_idx+2, base_idx))
    
    # Bottom fin
    base_idx += 3
    edges.append((base_idx, base_idx+1))
    edges.append((base_idx+1, base_idx+2))
    edges.append((base_idx+2, base_idx))
    
    # Right fin
    base_idx += 3
    edges.append((base_idx, base_idx+1))
    edges.append((base_idx+1, base_idx+2))
    edges.append((base_idx+2, base_idx))
    
    # Left fin
    base_idx += 3
    edges.append((base_idx, base_idx+1))
    edges.append((base_idx+1, base_idx+2))
    edges.append((base_idx+2, base_idx))
    
    return np.array(vertices), edges

def main():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D Submarine Simulator")
    clock = pygame.time.Clock()
    
    # Initialize vehicle
    # Control system "stepInput" is used but we will override u_control
    vehicle = torpedo(controlSystem="stepInput", r_rpm=0) 
    
    # State
    eta = np.zeros(6) # [x, y, z, phi, theta, psi]
    nu = np.zeros(6)  # [u, v, w, p, q, r]
    u_actual = np.zeros(vehicle.dimU)
    u_control = np.zeros(vehicle.dimU) # [top, bottom, star, port, rpm]
    
    # Initial position (slightly under water)
    eta[2] = 5.0
    
    # Mesh
    mesh_verts, mesh_edges = create_submarine_mesh()
    
    # Camera
    cam = Camera(width, height)
    
    # Input state
    target_rpm = 0.0
    rudder_angle = 0.0 # Yaw
    stern_angle = 0.0 # Pitch
    
    # Camera orbital state (Relative to vehicle)
    cam_dist = 6.0
    cam_yaw_rel = math.pi # 180 deg (Behind)
    cam_pitch_rel = -0.3  # Slightly above
    
    running = True
    pygame.mouse.get_rel() # Reset relative mouse
    
    while running:
        dt_ms = clock.tick(30)
        dt = dt_ms / 1000.0
        if dt > 0.1: dt = 0.1 # Clamp
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Mouse Camera Control
        if pygame.mouse.get_pressed()[2]: # Right Click held
            mx, my = pygame.mouse.get_rel()
            sensitivity = 0.005
            cam_yaw_rel += mx * sensitivity
            cam_pitch_rel += my * sensitivity
            
            # Clamp pitch
            cam_pitch_rel = max(min(cam_pitch_rel, 1.5), -1.5)
        else:
            pygame.mouse.get_rel() # Discard movement if not holding right click
                
        # Handle Keys
        keys = pygame.key.get_pressed()
        
        # RPM Control (Up/Down)
        if keys[pygame.K_UP]:
            target_rpm += 10
        if keys[pygame.K_DOWN]:
            target_rpm -= 10
            
        # Clamp RPM
        if target_rpm > 1525: target_rpm = 1525
        if target_rpm < -1525: target_rpm = -1525
        
        # Rudder/Stern Control (WASD)
        # W/S: Pitch (Stern planes)
        # A/D: Yaw (Rudders)
        
        # Reset angles
        # rudder_angle = 0.0
        # stern_angle = 0.0
        
        max_angle = 20 * math.pi / 180 # 20 degrees
        angle_step = 0.5 * math.pi / 180 # 0.5 degrees per frame
        
        if keys[pygame.K_w]: # Dive / Pitch Down
            stern_angle += angle_step
        if keys[pygame.K_s]: # Surface / Pitch Up
            stern_angle -= angle_step
            
        if keys[pygame.K_a]: # Turn Left
            rudder_angle += angle_step
        if keys[pygame.K_d]: # Turn Right
            rudder_angle -= angle_step
            
        # Clamp angles
        stern_angle = max(min(stern_angle, max_angle), -max_angle)
        rudder_angle = max(min(rudder_angle, max_angle), -max_angle)
        
        # Map to u_control
        # u_control indices:
        # 0: Top Rudder (270) -> Turn Right if +, Left if -? 
        #    Check torpedo logic: 
        #    Right Turn (D): Top +, Bottom -
        #    Left Turn (A): Top -, Bottom +
        #    My var 'rudder_angle' is + for Left (A).
        #    So A (Left): Top -, Bottom +. 
        #    So Top = -rudder_angle, Bottom = rudder_angle.
        
        # 2: Starboard Stern (180) -> Pitch Up if +?
        #    Pitch Up (S): Starboard +, Port -
        #    My var 'stern_angle' is - for Up (S).
        #    So S (Up): Starboard +, Port -. 
        #    Wait, if stern_angle is negative for S, then:
        #    Starboard = -stern_angle, Port = stern_angle.
        
        # Let's re-verify:
        # S pressed -> stern_angle = -max
        # We want Pitch Up. Pitch Up needs Starboard +, Port -.
        # So Starboard = -stern_angle (becomes +), Port = stern_angle (becomes -). Correct.
        
        # W pressed -> stern_angle = +max
        # We want Pitch Down. Pitch Down needs Starboard -, Port +.
        # Starboard = -stern_angle (-), Port = stern_angle (+). Correct.
        
        # A pressed -> rudder_angle = +max
        # We want Turn Left. Left needs Top -, Bottom +.
        # Top = -rudder_angle (-), Bottom = rudder_angle (+). Correct.
        
        # D pressed -> rudder_angle = -max
        # We want Turn Right. Right needs Top +, Bottom -.
        # Top = -rudder_angle (+), Bottom = rudder_angle (-). Correct.

        u_control[0] = -rudder_angle # Top
        u_control[1] = rudder_angle  # Bottom
        u_control[2] = -stern_angle  # Starboard
        u_control[3] = stern_angle   # Port
        u_control[4] = target_rpm
        
        # Physics Step
        nu, u_actual = vehicle.dynamics(eta, nu, u_actual, u_control, dt)
        eta = attitudeEuler(eta, nu, dt)
        
        # --- Rendering ---
        screen.fill(BLACK) # Underwater color? Dark Blue?
        screen.fill((0, 20, 40))
        
        # Update Camera
        # Calculate camera position based on orbital parameters relative to vehicle body
        
        # 1. Calculate camera position in Body Frame (Spherical -> Cartesian)
        # Note: In body frame, X is forward, Z is down.
        # We want spherical coords where yaw=0 is forward (+X), pitch=0 is horizontal.
        # x = r * cos(pitch) * cos(yaw)
        # y = r * cos(pitch) * sin(yaw)
        # z = -r * sin(pitch)  (Negative because Z is down and pitch up is usually +Z in math, but here we want visual up)
        
        cx_b = cam_dist * math.cos(cam_pitch_rel) * math.cos(cam_yaw_rel)
        cy_b = cam_dist * math.cos(cam_pitch_rel) * math.sin(cam_yaw_rel)
        cz_b = -cam_dist * math.sin(cam_pitch_rel) 
        
        cam_offset_body = np.array([cx_b, cy_b, cz_b])
        
        # 2. Transform to World Frame
        R_vehicle = Rzyx(eta[3], eta[4], eta[5])
        cam_pos_world = eta[0:3] + R_vehicle @ cam_offset_body
        
        # Smooth camera follow (optional)
        # For now, hard attach
        cam.pos = cam_pos_world
        
        # Look at sub
        cam.target = eta[0:3]
        
        # Prepare vertices
        # Rotate and Translate
        transformed_verts = []
        for v in mesh_verts:
            v_world = eta[0:3] + R_vehicle @ v
            transformed_verts.append(v_world)
        
        transformed_verts = np.array(transformed_verts)
        
        # Project
        projected_points = cam.project(transformed_verts)
        
        # Draw edges
        for i, j in mesh_edges:
            p1 = projected_points[i]
            p2 = projected_points[j]
            if p1 and p2:
                pygame.draw.line(screen, GREEN, p1, p2, 2)
                
        # Draw Water Surface (Grid)
        # Simple grid at z=0
        # Draw only lines near the sub
        grid_size = 20
        grid_step = 2
        sub_x, sub_y = eta[0], eta[1]
        start_x = int(sub_x / grid_step) * grid_step - grid_size
        start_y = int(sub_y / grid_step) * grid_step - grid_size
        
        surface_points = []
        # Create grid lines
        for x in range(int(start_x), int(start_x + 2 * grid_size), grid_step):
            for y in range(int(start_y), int(start_y + 2 * grid_size), grid_step):
                # We need lines.
                # Just horizontal and vertical lines
                pass
                
        # Better: just draw lines
        # X-lines
        for i in range(-10, 11):
            x = int(sub_x / grid_step) * grid_step + i * grid_step
            p_start = np.array([x, sub_y - 20, 0])
            p_end = np.array([x, sub_y + 20, 0])
            pts = cam.project(np.array([p_start, p_end]))
            if pts[0] and pts[1]:
                pygame.draw.line(screen, (0, 100, 200), pts[0], pts[1], 1)
                
        # Y-lines
        for i in range(-10, 11):
            y = int(sub_y / grid_step) * grid_step + i * grid_step
            p_start = np.array([sub_x - 20, y, 0])
            p_end = np.array([sub_x + 20, y, 0])
            pts = cam.project(np.array([p_start, p_end]))
            if pts[0] and pts[1]:
                pygame.draw.line(screen, (0, 100, 200), pts[0], pts[1], 1)
        
        # Info Overlay
        font = pygame.font.SysFont("Arial", 18)
        
        # Convert rad to deg for display
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
            f"Rudders (Top/Bot): {top_deg:.1f} / {bot_deg:.1f} deg",
            f"Sterns (Stb/Prt): {stb_deg:.1f} / {prt_deg:.1f} deg",
            "Controls: WASD + Arrow Up/Down"
        ]
        
        for i, text in enumerate(infos):
            surf = font.render(text, True, WHITE)
            screen.blit(surf, (10, 10 + i * 20))
            
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()
