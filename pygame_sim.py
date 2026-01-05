import pygame
import numpy as np
import math
import sys
from torpedo import torpedo
from lib.gnc import attitudeEuler, Rzyx

# 颜色
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
        self.pos = np.array([-5.0, 0.0, -2.0]) # 初始相机位置（位于后方和上方）
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 0.0, -1.0]) # NED坐标系：z轴向下，所以-z是向上
        self.fov = 60
        self.aspect = width / height
        self.near = 0.1
        self.far = 100.0

    def get_view_matrix(self):
        # 简单的注视矩阵
        z_axis = self.target - self.pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(z_axis, self.up)
        if np.linalg.norm(x_axis) < 1e-6:
             x_axis = np.array([1.0, 0.0, 0.0]) # 备用
        else:
             x_axis = x_axis / np.linalg.norm(x_axis)
             
        y_axis = np.cross(x_axis, z_axis)
        
        # 视图矩阵（世界 -> 相机）
        # R = [x_axis, y_axis, z_axis]^T
        R = np.array([x_axis, y_axis, z_axis])
        t = -R @ self.pos
        
        V = np.identity(4)
        V[:3, :3] = R
        V[:3, 3] = t
        return V

    def project(self, points_3d):
        # points_3d: N x 3
        # 变换到相机空间
        view = self.get_view_matrix()
        
        # 齐次坐标
        ones = np.ones((len(points_3d), 1))
        points_4d = np.hstack([points_3d, ones])
        
        # 相机空间
        cam_points = points_4d @ view.T
        
        # 投影到屏幕
        # 我们在相机空间中沿+Z方向看？
        # 等等，我的 get_view_matrix 构造的 Z 指向目标。
        # 通常 OpenGL 相机沿 -Z 方向看。
        # 让我们调整一下：Z 应该是前方。
        # z_axis = target - pos（前方）
        
        # 透视投影
        # x' = x / z * f
        # y' = y / z * f
        
        # 简单投影
        f = self.width / (2 * math.tan(math.radians(self.fov) / 2))
        
        projected = []
        for p in cam_points:
            x, y, z, w = p
            # 检查点是否在相机后面
            if z <= self.near: 
                projected.append(None)
                continue
                
            px = x * f / z + self.width / 2
            py = -y * f / z + self.height / 2 # 翻转 Y 轴以适应屏幕坐标
            projected.append((px, py))
            
        return projected

def create_submarine_mesh():
    # 简单的圆柱体 + 圆锥体
    vertices = []
    edges = []
    
    # 艇身（圆柱体）
    segments = 8
    radius = 0.2
    length = 1.6
    x_front = length / 2
    x_back = -length / 2
    
    # 前圆
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        y = radius * math.cos(angle)
        z = radius * math.sin(angle)
        vertices.append([x_front, y, z])
        
    # 后圆
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        y = radius * math.cos(angle)
        z = radius * math.sin(angle)
        vertices.append([x_back, y, z])
        
    # 艇首（圆锥体）
    vertices.append([x_front + 0.3, 0, 0])
    nose_idx = len(vertices) - 1
    
    # 连接边
    for i in range(segments):
        # 前圆
        edges.append((i, (i + 1) % segments))
        # 后圆
        edges.append((segments + i, segments + (i + 1) % segments))
        # 连接线
        edges.append((i, segments + i))
        # 艇首
        edges.append((i, nose_idx))
        
    # 舵翼（简单的三角形）
    # 上舵
    vertices.append([x_back, 0, -radius]) # 基底
    vertices.append([x_back - 0.2, 0, -radius]) # 基底后部
    vertices.append([x_back - 0.1, 0, -radius - 0.3]) # 尖端
    
    # 下舵
    vertices.append([x_back, 0, radius])
    vertices.append([x_back - 0.2, 0, radius])
    vertices.append([x_back - 0.1, 0, radius + 0.3])
    
    # 右艉舵（右舷）- Y 是右边？
    # NED：x 北，y 东（右），z 下。
    vertices.append([x_back, radius, 0])
    vertices.append([x_back - 0.2, radius, 0])
    vertices.append([x_back - 0.1, radius + 0.3, 0])
    
    # 左艉舵（左舷）
    vertices.append([x_back, -radius, 0])
    vertices.append([x_back - 0.2, -radius, 0])
    vertices.append([x_back - 0.1, -radius - 0.3, 0])
    
    # 手动添加舵翼边缘还是直接将所有顶点画成线？
    # 我们只画所有定义的边加上舵翼边
    base_idx = nose_idx + 1
    # 上舵翼
    edges.append((base_idx, base_idx+1))
    edges.append((base_idx+1, base_idx+2))
    edges.append((base_idx+2, base_idx))
    
    # 下舵翼
    base_idx += 3
    edges.append((base_idx, base_idx+1))
    edges.append((base_idx+1, base_idx+2))
    edges.append((base_idx+2, base_idx))
    
    # 右舵翼
    base_idx += 3
    edges.append((base_idx, base_idx+1))
    edges.append((base_idx+1, base_idx+2))
    edges.append((base_idx+2, base_idx))
    
    # 左舵翼
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
    
    # 初始化潜航器
    # 使用 "stepInput" 控制系统，但我们将覆盖 u_control
    vehicle = torpedo(controlSystem="stepInput", r_rpm=0) 
    
    # 状态
    eta = np.zeros(6) # [x, y, z, phi, theta, psi]
    nu = np.zeros(6)  # [u, v, w, p, q, r]
    u_actual = np.zeros(vehicle.dimU)
    u_control = np.zeros(vehicle.dimU) # [top, bottom, star, port, rpm]
    
    # 初始位置（略微在水下）
    eta[2] = 5.0
    
    # 网格
    mesh_verts, mesh_edges = create_submarine_mesh()
    
    # 相机
    cam = Camera(width, height)
    
    # 输入状态
    target_rpm = 0.0
    rudder_angle = 0.0 # 偏航
    stern_angle = 0.0 # 俯仰
    
    # 每个舵翼的微调 [上, 下, 右, 左]
    fin_offsets = np.zeros(4)
    
    # 相机轨道状态（相对于潜航器）
    cam_dist = 6.0
    cam_yaw_rel = math.pi # 180 度（后方）
    cam_pitch_rel = -0.3  # 略微上方
    
    running = True
    pygame.mouse.get_rel() # 重置相对鼠标
    
    while running:
        dt_ms = clock.tick(30)
        dt = dt_ms / 1000.0
        if dt > 0.1: dt = 0.1 # 限制范围
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 鼠标相机控制
        if pygame.mouse.get_pressed()[2]: # 按住右键
            mx, my = pygame.mouse.get_rel()
            sensitivity = 0.005
            cam_yaw_rel += mx * sensitivity
            cam_pitch_rel += my * sensitivity
            
            # 限制俯仰角
            cam_pitch_rel = max(min(cam_pitch_rel, 1.5), -1.5)
        else:
            pygame.mouse.get_rel() # 如果未按住右键，则丢弃移动
                
        # 处理按键
        keys = pygame.key.get_pressed()
        
        # 转速控制（上/下）
        if keys[pygame.K_UP]:
            target_rpm += 10
        if keys[pygame.K_DOWN]:
            target_rpm -= 10
            
        # 限制转速
        if target_rpm > 1525: target_rpm = 1525
        if target_rpm < -1525: target_rpm = -1525
        
        # 舵/艉舵控制 (WASD)
        # W/S: 俯仰（艉舵）
        # A/D: 偏航（方向舵）
        
        # 重置角度
        # rudder_angle = 0.0
        # stern_angle = 0.0
        
        max_angle = 20 * math.pi / 180 # 20 度
        angle_step = 0.5 * math.pi / 180 # 每帧 0.5 度
        
        if keys[pygame.K_w]: # 下潜 / 俯仰向下
            stern_angle += angle_step
        if keys[pygame.K_s]: # 上浮 / 俯仰向上
            stern_angle -= angle_step
            
        if keys[pygame.K_a]: # 左转
            rudder_angle += angle_step
        if keys[pygame.K_d]: # 右转
            rudder_angle -= angle_step
            
        # 单独舵翼微调 (1, 2, 3, 4) + Shift 减小
        fine_step = 0.1 * math.pi / 180
        direction = -1 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1
        
        if keys[pygame.K_1]: fin_offsets[0] += direction * fine_step # 上
        if keys[pygame.K_2]: fin_offsets[1] += direction * fine_step # 下
        if keys[pygame.K_3]: fin_offsets[2] += direction * fine_step # 右
        if keys[pygame.K_4]: fin_offsets[3] += direction * fine_step # 左
            
        # 限制角度
        stern_angle = max(min(stern_angle, max_angle), -max_angle)
        rudder_angle = max(min(rudder_angle, max_angle), -max_angle)
        
        # 映射到 u_control
        # u_control 索引：
        # 0: 上舵 (270) -> 正值右转，负值左转？
        #    检查鱼雷逻辑：
        #    右转 (D): 上舵 +, 下舵 -
        #    左转 (A): 上舵 -, 下舵 +
        #    我的变量 'rudder_angle' 左转 (A) 为正。
        #    所以 A (左): 上舵 -, 下舵 +。
        #    所以上舵 = -rudder_angle, 下舵 = rudder_angle。
        
        # 2: 右艉舵 (180) -> 正值俯仰向上？
        #    俯仰向上 (S): 右舷 +, 左舷 -
        #    我的变量 'stern_angle' 向上 (S) 为负。
        #    所以 S (向上): 右舷 +, 左舷 -。
        #    等等，如果 stern_angle 对于 S 是负的，那么：
        #    右舷 = -stern_angle, 左舷 = stern_angle。
        
        # 让我们重新验证：
        # 按下 S -> stern_angle = -max
        # 我们想要俯仰向上。俯仰向上需要右舷 +, 左舷 -。
        # 所以右舷 = -stern_angle (变正), 左舷 = stern_angle (变负)。正确。
        
        # 按下 W -> stern_angle = +max
        # 我们想要俯仰向下。俯仰向下需要右舷 -, 左舷 +。
        # 右舷 = -stern_angle (-), 左舷 = stern_angle (+)。正确。
        
        # 按下 A -> rudder_angle = +max
        # 我们想要左转。左转需要上舵 -, 下舵 +。
        # 上舵 = -rudder_angle (-), 下舵 = rudder_angle (+)。正确。
        
        # 按下 D -> rudder_angle = -max
        # 我们想要右转。右转需要上舵 +, 下舵 -。
        # 上舵 = -rudder_angle (+), 下舵 = rudder_angle (-)。正确。

        u_control[0] = -rudder_angle + fin_offsets[0] # 上
        u_control[1] = rudder_angle + fin_offsets[1]  # 下
        u_control[2] = -stern_angle + fin_offsets[2]  # 右
        u_control[3] = stern_angle + fin_offsets[3]   # 左
        u_control[4] = target_rpm
        
        # 物理步进
        nu, u_actual = vehicle.dynamics(eta, nu, u_actual, u_control, dt)
        eta = attitudeEuler(eta, nu, dt)
        
        # --- 渲染 ---
        screen.fill(BLACK) # 水下颜色？深蓝？
        screen.fill((0, 20, 40))
        
        # 更新相机
        # 根据相对于潜航器本体的轨道参数计算相机位置
        
        # 1. 计算本体坐标系中的相机位置（球坐标 -> 笛卡尔坐标）
        # 注意：在本体坐标系中，X 是前方，Z 是下方。
        # 我们想要球坐标，其中 yaw=0 是前方 (+X)，pitch=0 是水平。
        # x = r * cos(pitch) * cos(yaw)
        # y = r * cos(pitch) * sin(yaw)
        # z = -r * sin(pitch) （负值是因为 Z 向下，而数学上俯仰向上通常是 +Z，但这里我们想要视觉上的向上）
        
        cx_b = cam_dist * math.cos(cam_pitch_rel) * math.cos(cam_yaw_rel)
        cy_b = cam_dist * math.cos(cam_pitch_rel) * math.sin(cam_yaw_rel)
        cz_b = -cam_dist * math.sin(cam_pitch_rel) 
        
        cam_offset_body = np.array([cx_b, cy_b, cz_b])
        
        # 2. 变换到世界坐标系
        R_vehicle = Rzyx(eta[3], eta[4], eta[5])
        cam_pos_world = eta[0:3] + R_vehicle @ cam_offset_body
        
        # 平滑相机跟随（可选）
        # 目前硬连接
        cam.pos = cam_pos_world
        
        # 注视潜航器
        cam.target = eta[0:3]
        
        # 准备顶点
        # 旋转和平移
        transformed_verts = []
        for v in mesh_verts:
            v_world = eta[0:3] + R_vehicle @ v
            transformed_verts.append(v_world)
        
        transformed_verts = np.array(transformed_verts)
        
        # 投影
        projected_points = cam.project(transformed_verts)
        
        # 绘制边
        for i, j in mesh_edges:
            p1 = projected_points[i]
            p2 = projected_points[j]
            if p1 and p2:
                pygame.draw.line(screen, GREEN, p1, p2, 2)
                
        # 绘制水面（网格）
        # z=0 处的简单网格
        # 只绘制潜航器附近的线
        grid_size = 20
        grid_step = 2
        sub_x, sub_y = eta[0], eta[1]
        start_x = int(sub_x / grid_step) * grid_step - grid_size
        start_y = int(sub_y / grid_step) * grid_step - grid_size
        
        surface_points = []
        # 创建网格线
        for x in range(int(start_x), int(start_x + 2 * grid_size), grid_step):
            for y in range(int(start_y), int(start_y + 2 * grid_size), grid_step):
                # 我们需要线。
                # 只是水平和垂直线
                pass
                
        # 更好：直接画线
        # X方向线
        for i in range(-10, 11):
            x = int(sub_x / grid_step) * grid_step + i * grid_step
            p_start = np.array([x, sub_y - 20, 0])
            p_end = np.array([x, sub_y + 20, 0])
            pts = cam.project(np.array([p_start, p_end]))
            if pts[0] and pts[1]:
                pygame.draw.line(screen, (0, 100, 200), pts[0], pts[1], 1)
                
        # Y方向线
        for i in range(-10, 11):
            y = int(sub_y / grid_step) * grid_step + i * grid_step
            p_start = np.array([sub_x - 20, y, 0])
            p_end = np.array([sub_x + 20, y, 0])
            pts = cam.project(np.array([p_start, p_end]))
            if pts[0] and pts[1]:
                pygame.draw.line(screen, (0, 100, 200), pts[0], pts[1], 1)
        
        # 信息覆盖
        font = pygame.font.SysFont("Arial", 18)
        
        # 转换为度数以显示
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
            "Controls: WASD + Arrow Up/Down",
            "Fine Tune: 1,2,3,4 (+Shift)"
        ]
        
        for i, text in enumerate(infos):
            surf = font.render(text, True, WHITE)
            screen.blit(surf, (10, 10 + i * 20))
            
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()
