import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from env_cfg import EnvConfig


@dataclass
class DynamicObstacle:
    """
    动态障碍：用圆形表示，速度恒定，发生碰撞时反弹
    """
    x: float
    y: float
    vx: float
    vy: float
    radius: float


class CleaningRobotEnv(gym.Env):
    """
    二维连续扫地机器人多任务环境

    本版本相较原版重点改动：
    1. state 显式增强：
       - dist_to_dock_norm
       - dock_angle_error
       - front_min_ray_dist
       - min_ray_dist
       - left_mean_ray_dist
       - right_mean_ray_dist
       - front_has_dynamic
       - battery_can_return_hint

    2. local patch 显式加入 dock 通道（可开关）
    3. 增加地图连通性 / dock 可达性约束
    4. 增加 training_mode（baseline_easy / mid / full）
    5. reward 精简，并整体 reward_scale 缩放
    6. step/info 中稳定输出 done_reason
    7. 新增反挂机机制：
       - 长时间原地不动惩罚
       - 长时间无清扫进展惩罚
       - 高电量占据充电桩惩罚
       - 在充电桩长时间静止惩罚
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()

        self.cfg = cfg if cfg is not None else EnvConfig()
        self.render_mode = render_mode

        self.W = self.cfg.map_cfg.width
        self.H = self.cfg.map_cfg.height
        self.grid = self.cfg.map_cfg.clean_grid_size
        self.grid_w = self.W // self.grid
        self.grid_h = self.H // self.grid

        self.dt = self.cfg.dt
        self.max_steps = self.cfg.max_steps

        self._np_random: Optional[np.random.Generator] = None

        # 根据训练阶段对部分配置进行轻量重映射
        self._stage_dynamic_enabled = self.cfg.dynamic_obstacle.enabled
        self._stage_battery_capacity = self.cfg.robot.battery_capacity
        self._stage_obstacle_count_range = self.cfg.map_cfg.obstacle_count_range
        self._apply_training_mode()

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # state 维度：
        # [x/W, y/H, cos(theta), sin(theta),
        #  v/vmax, w/wmax,
        #  battery_ratio, coverage,
        #  dock_dx_norm, dock_dy_norm,
        #  is_low_battery, on_dock,
        #  dist_to_dock_norm, dock_angle_error,
        #  front_min_ray_dist, min_ray_dist,
        #  left_mean_ray_dist, right_mean_ray_dist,
        #  front_has_dynamic, battery_can_return_hint]
        self.state_dim = 20

        local_size = self.cfg.sensor.local_map_size
        ray_shape = (self.cfg.sensor.num_rays, 3)

        if self.cfg.use_dict_observation:
            obs_dict: Dict[str, spaces.Space] = {
                "state": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
                ),
                "rays": spaces.Box(
                    low=0.0, high=1.0, shape=ray_shape, dtype=np.float32
                ),
                "local_clean_map": spaces.Box(
                    low=0.0, high=1.0, shape=(local_size, local_size), dtype=np.float32
                ),
                "local_obstacle_map": spaces.Box(
                    low=0.0, high=1.0, shape=(local_size, local_size), dtype=np.float32
                ),
            }

            if self.cfg.sensor.use_local_dock_map:
                obs_dict["local_dock_map"] = spaces.Box(
                    low=0.0, high=1.0, shape=(local_size, local_size), dtype=np.float32
                )

            self.observation_space = spaces.Dict(obs_dict)
        else:
            flat_dim = (
                self.state_dim
                + self.cfg.sensor.num_rays * 3
                + local_size * local_size
                + local_size * local_size
            )
            if self.cfg.sensor.use_local_dock_map:
                flat_dim += local_size * local_size

            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(flat_dim,), dtype=np.float32
            )

        # 环境状态
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.battery = 0.0

        self.steps = 0
        self.clean_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.obstacle_map = np.zeros(
            (self.grid_h, self.grid_w), dtype=np.float32)
        self.dock_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        self.total_cleanable_cells = 0
        self.reachable_from_dock_map = np.zeros(
            (self.grid_h, self.grid_w), dtype=np.float32)

        self.obstacles: List[pygame.Rect] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []

        self.dock_rect: Optional[pygame.Rect] = None

        self.prev_coverage = 0.0
        self.prev_dist_to_dock = 0.0
        self.prev_on_dock = False
        self.low_battery_entered = False
        self.last_done_reason: Optional[str] = None

        self.last_ray_hits: Optional[np.ndarray] = None
        self.last_ray_summary: Dict[str, float] = {}

        # 反挂机计数器
        self.idle_steps = 0
        self.no_progress_steps = 0
        self.dock_idle_steps = 0

        # pygame
        self.window = None
        self.clock = None
        self.canvas = None

    # =========================================================
    # 训练阶段
    # =========================================================
    def _apply_training_mode(self) -> None:
        """
        根据 training_mode 对环境难度做分阶段调整。
        """
        mode = self.cfg.training_mode

        if mode == "baseline_easy":
            self._stage_dynamic_enabled = self.cfg.curriculum.easy_dynamic_enabled
            self._stage_battery_capacity = self.cfg.robot.battery_capacity * \
                self.cfg.curriculum.easy_battery_capacity_scale
            scale = self.cfg.curriculum.easy_obstacle_scale
        elif mode == "baseline_mid":
            self._stage_dynamic_enabled = self.cfg.curriculum.mid_dynamic_enabled
            self._stage_battery_capacity = self.cfg.robot.battery_capacity * \
                self.cfg.curriculum.mid_battery_capacity_scale
            scale = self.cfg.curriculum.mid_obstacle_scale
        else:
            self._stage_dynamic_enabled = self.cfg.curriculum.full_dynamic_enabled and self.cfg.dynamic_obstacle.enabled
            self._stage_battery_capacity = self.cfg.robot.battery_capacity * \
                self.cfg.curriculum.full_battery_capacity_scale
            scale = self.cfg.curriculum.full_obstacle_scale

        lo, hi = self.cfg.map_cfg.obstacle_count_range
        scaled_lo = max(0, int(round(lo * scale)))
        scaled_hi = max(scaled_lo, int(round(hi * scale)))
        self._stage_obstacle_count_range = (scaled_lo, scaled_hi)

    # =========================================================
    # 基础工具
    # =========================================================
    def _seed(self, seed=None) -> None:
        self._np_random = np.random.default_rng(seed)

    def _normalize_angle(self, angle: float) -> float:
        return ((angle + math.pi) % (2 * math.pi)) - math.pi

    def _circle_rect_collision(self, cx: float, cy: float, radius: float, rect: pygame.Rect) -> bool:
        closest_x = np.clip(cx, rect.left, rect.right)
        closest_y = np.clip(cy, rect.top, rect.bottom)
        dx = cx - closest_x
        dy = cy - closest_y
        return dx * dx + dy * dy <= radius * radius

    def _circle_circle_collision(self, x1: float, y1: float, r1: float, x2: float, y2: float, r2: float) -> bool:
        dx = x1 - x2
        dy = y1 - y2
        return dx * dx + dy * dy <= (r1 + r2) * (r1 + r2)

    def _build_dock(self) -> None:
        s = self.cfg.map_cfg.dock_size
        x, y = self.cfg.map_cfg.dock_pos
        self.dock_rect = pygame.Rect(x, y, s, s)

    def _build_dock_map(self) -> None:
        """
        将 dock 区域映射到网格上，供 local_dock_map 使用。
        """
        self.dock_map.fill(0.0)

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                cell_rect = pygame.Rect(
                    gx * self.grid, gy * self.grid, self.grid, self.grid)
                if cell_rect.colliderect(self.dock_rect):
                    self.dock_map[gy, gx] = 1.0

    def _dist_to_dock(self, x: float, y: float) -> float:
        cx = self.dock_rect.centerx
        cy = self.dock_rect.centery
        return math.hypot(x - cx, y - cy)

    def _dock_direction(self, x: float, y: float) -> Tuple[float, float]:
        cx = self.dock_rect.centerx
        cy = self.dock_rect.centery
        dx = cx - x
        dy = cy - y
        return dx, dy

    def _robot_on_dock(self, x: Optional[float] = None, y: Optional[float] = None) -> bool:
        if x is None:
            x = self.robot_x
        if y is None:
            y = self.robot_y

        robot_rect = pygame.Rect(
            int(x - self.cfg.robot.radius),
            int(y - self.cfg.robot.radius),
            int(self.cfg.robot.radius * 2),
            int(self.cfg.robot.radius * 2),
        )
        return robot_rect.colliderect(self.dock_rect)

    def _point_hits_static(self, x: float, y: float) -> bool:
        bt = self.cfg.map_cfg.border_thickness
        if x < bt or x > self.W - bt or y < bt or y > self.H - bt:
            return True
        for rect in self.obstacles:
            if rect.collidepoint(x, y):
                return True
        return False

    def _point_hits_dynamic(self, x: float, y: float) -> bool:
        for obs in self.dynamic_obstacles:
            dx = x - obs.x
            dy = y - obs.y
            if dx * dx + dy * dy <= obs.radius * obs.radius:
                return True
        return False

    def _robot_collides_static(self, x: float, y: float) -> bool:
        r = self.cfg.robot.radius
        bt = self.cfg.map_cfg.border_thickness

        if x - r < bt:
            return True
        if x + r > self.W - bt:
            return True
        if y - r < bt:
            return True
        if y + r > self.H - bt:
            return True

        for rect in self.obstacles:
            if self._circle_rect_collision(x, y, r, rect):
                return True
        return False

    def _robot_collides_dynamic(self, x: float, y: float) -> bool:
        r = self.cfg.robot.radius
        for obs in self.dynamic_obstacles:
            if self._circle_circle_collision(x, y, r, obs.x, obs.y, obs.radius):
                return True
        return False

    def _robot_collides_any(self, x: float, y: float) -> Tuple[bool, bool]:
        coll_static = self._robot_collides_static(x, y)
        coll_dynamic = self._robot_collides_dynamic(x, y)
        return coll_static, coll_dynamic

    def _cell_is_free(self, gx: int, gy: int) -> bool:
        if not (0 <= gx < self.grid_w and 0 <= gy < self.grid_h):
            return False
        return self.obstacle_map[gy, gx] < 0.5

    def _dock_grid_center(self) -> Tuple[int, int]:
        gx = int(self.dock_rect.centerx // self.grid)
        gy = int(self.dock_rect.centery // self.grid)
        gx = int(np.clip(gx, 0, self.grid_w - 1))
        gy = int(np.clip(gy, 0, self.grid_h - 1))
        return gx, gy

    def _compute_reachable_from_dock(self) -> np.ndarray:
        """
        计算从 dock 所在连通块可到达的 free cell。
        """
        reachable = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        start_gx, start_gy = self._dock_grid_center()

        if not self._cell_is_free(start_gx, start_gy):
            return reachable

        q = deque()
        q.append((start_gx, start_gy))
        reachable[start_gy, start_gx] = 1.0

        while q:
            gx, gy = q.popleft()
            for nx, ny in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    if reachable[ny, nx] > 0.5:
                        continue
                    if self._cell_is_free(nx, ny):
                        reachable[ny, nx] = 1.0
                        q.append((nx, ny))

        return reachable

    def _map_is_valid(self) -> bool:
        """
        地图有效性检查：
        1. 可清扫比例足够
        2. 障碍比例不过高
        3. dock 所在连通块占比足够
        """
        free_ratio = float((self.obstacle_map < 0.5).sum()) / \
            float(self.grid_h * self.grid_w)
        obstacle_ratio = 1.0 - free_ratio

        if free_ratio < self.cfg.map_cfg.min_cleanable_ratio:
            return False
        if obstacle_ratio > self.cfg.map_cfg.max_obstacle_ratio:
            return False

        if self.cfg.map_cfg.require_dock_reachable_component:
            reachable = self._compute_reachable_from_dock()
            reachable_free = float((reachable > 0.5).sum())
            total_free = float((self.obstacle_map < 0.5).sum())
            if total_free <= 0:
                return False
            ratio = reachable_free / total_free
            if ratio < self.cfg.map_cfg.min_reachable_ratio_from_dock:
                return False

        return True

    def _generate_random_obstacles(self) -> None:
        self.obstacles = []

        count = self._np_random.integers(
            self._stage_obstacle_count_range[0],
            self._stage_obstacle_count_range[1] + 1
        )

        min_w, min_h = self.cfg.map_cfg.obstacle_min_size
        max_w, max_h = self.cfg.map_cfg.obstacle_max_size
        margin = self.cfg.map_cfg.obstacle_margin

        dock_avoid = self.dock_rect.inflate(
            self.cfg.map_cfg.dock_safe_clearance * 2,
            self.cfg.map_cfg.dock_safe_clearance * 2
        )

        for _ in range(count):
            placed = False
            for _ in range(200):
                w = int(self._np_random.integers(min_w, max_w + 1))
                h = int(self._np_random.integers(min_h, max_h + 1))
                x = int(self._np_random.integers(40, max(41, self.W - w - 40)))
                y = int(self._np_random.integers(40, max(41, self.H - h - 40)))
                rect = pygame.Rect(x, y, w, h)

                if rect.colliderect(dock_avoid):
                    continue

                overlap = False
                for other in self.obstacles:
                    if rect.inflate(margin, margin).colliderect(other.inflate(margin, margin)):
                        overlap = True
                        break
                if overlap:
                    continue

                self.obstacles.append(rect)
                placed = True
                break

            if not placed:
                continue

    def _build_obstacle_map(self) -> None:
        self.obstacle_map.fill(0.0)

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                cell_rect = pygame.Rect(
                    gx * self.grid, gy * self.grid, self.grid, self.grid)
                blocked = False
                for rect in self.obstacles:
                    if cell_rect.colliderect(rect):
                        blocked = True
                        break
                if blocked:
                    self.obstacle_map[gy, gx] = 1.0

        # dock 区域强制视为可通行
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                cell_rect = pygame.Rect(
                    gx * self.grid, gy * self.grid, self.grid, self.grid)
                if cell_rect.colliderect(self.dock_rect):
                    self.obstacle_map[gy, gx] = 0.0

        self.total_cleanable_cells = int((self.obstacle_map < 0.5).sum())

    def _build_valid_map(self, max_retry: int = 80) -> None:
        """
        反复采样地图，直到满足可清扫比例、障碍比例、dock 连通性等约束。
        """
        for _ in range(max_retry):
            self._generate_random_obstacles()
            self._build_obstacle_map()
            self._build_dock_map()

            if self._map_is_valid():
                self.reachable_from_dock_map = self._compute_reachable_from_dock()
                return

        self._build_obstacle_map()
        self._build_dock_map()
        self.reachable_from_dock_map = self._compute_reachable_from_dock()

    def _grid_reachable_to_dock(self, x: float, y: float) -> bool:
        gx = int(np.clip(int(x // self.grid), 0, self.grid_w - 1))
        gy = int(np.clip(int(y // self.grid), 0, self.grid_h - 1))
        return self.reachable_from_dock_map[gy, gx] > 0.5

    def _random_free_pose(self, max_try: int = 1000) -> Tuple[float, float, float]:
        """
        采样初始位姿。
        若要求 spawn 可达 dock，则只从 dock 可达连通块中采样。
        """
        for _ in range(max_try):
            x = self._np_random.uniform(100, self.W - 100)
            y = self._np_random.uniform(100, self.H - 100)
            theta = self._np_random.uniform(-math.pi, math.pi)

            if self._dist_to_dock(x, y) < self.cfg.robot.spawn_safe_radius:
                continue

            coll_s, coll_d = self._robot_collides_any(x, y)
            if coll_s or coll_d:
                continue

            if self.cfg.map_cfg.require_spawn_reachable_to_dock:
                if not self._grid_reachable_to_dock(x, y):
                    continue

            return x, y, theta

        ys, xs = np.where(self.reachable_from_dock_map > 0.5)
        if len(xs) > 0:
            idx = int(self._np_random.integers(0, len(xs)))
            gx = int(xs[idx])
            gy = int(ys[idx])
            x = gx * self.grid + self.grid * 0.5
            y = gy * self.grid + self.grid * 0.5
            return x, y, 0.0

        return self.W * 0.5, self.H * 0.5, 0.0

    # =========================================================
    # 动态障碍
    # =========================================================
    def _spawn_dynamic_obstacles(self) -> None:
        self.dynamic_obstacles = []

        if not self._stage_dynamic_enabled:
            return

        count = self._np_random.integers(
            self.cfg.dynamic_obstacle.count_range[0],
            self.cfg.dynamic_obstacle.count_range[1] + 1
        )

        r_min, r_max = self.cfg.dynamic_obstacle.radius_range
        v_min, v_max = self.cfg.dynamic_obstacle.speed_range

        for _ in range(count):
            placed = False
            for _ in range(self.cfg.dynamic_obstacle.max_try_spawn):
                radius = float(self._np_random.integers(r_min, r_max + 1))
                x = self._np_random.uniform(80, self.W - 80)
                y = self._np_random.uniform(80, self.H - 80)
                speed = self._np_random.uniform(v_min, v_max)
                angle = self._np_random.uniform(-math.pi, math.pi)
                vx = speed * math.cos(angle)
                vy = speed * math.sin(angle)

                if self._robot_collides_static(x, y):
                    continue
                if self._dist_to_dock(x, y) < self.cfg.dynamic_obstacle.dock_avoid_radius:
                    continue
                if self._grid_reachable_to_dock(x, y) is False:
                    continue

                ok = True
                for obs in self.dynamic_obstacles:
                    if self._circle_circle_collision(
                        x, y, radius,
                        obs.x, obs.y, obs.radius + self.cfg.dynamic_obstacle.pair_safe_gap
                    ):
                        ok = False
                        break
                if not ok:
                    continue

                self.dynamic_obstacles.append(DynamicObstacle(
                    x=x, y=y, vx=vx, vy=vy, radius=radius
                ))
                placed = True
                break

            if not placed:
                continue

    def _move_dynamic_obstacles(self) -> None:
        bt = self.cfg.map_cfg.border_thickness

        for obs in self.dynamic_obstacles:
            nx = obs.x + obs.vx * self.dt
            ny = obs.y + obs.vy * self.dt

            bounced = False

            if nx - obs.radius < bt or nx + obs.radius > self.W - bt:
                obs.vx *= -1.0
                bounced = True
            if ny - obs.radius < bt or ny + obs.radius > self.H - bt:
                obs.vy *= -1.0
                bounced = True

            if bounced:
                nx = obs.x + obs.vx * self.dt
                ny = obs.y + obs.vy * self.dt

            hit_static = False
            for rect in self.obstacles:
                if self._circle_rect_collision(nx, ny, obs.radius, rect):
                    hit_static = True
                    break

            if self._circle_rect_collision(nx, ny, obs.radius, self.dock_rect):
                hit_static = True

            if hit_static:
                obs.vx *= -1.0
                obs.vy *= -1.0
                nx = obs.x + obs.vx * self.dt
                ny = obs.y + obs.vy * self.dt

            for other in self.dynamic_obstacles:
                if other is obs:
                    continue
                if self._circle_circle_collision(nx, ny, obs.radius, other.x, other.y, other.radius):
                    obs.vx *= -1.0
                    obs.vy *= -1.0
                    nx = obs.x + obs.vx * self.dt
                    ny = obs.y + obs.vy * self.dt
                    break

            obs.x = float(np.clip(nx, bt + obs.radius,
                          self.W - bt - obs.radius))
            obs.y = float(np.clip(ny, bt + obs.radius,
                          self.H - bt - obs.radius))

    # =========================================================
    # 清扫 / 观测
    # =========================================================
    def _mark_cleaned(self) -> Tuple[int, int]:
        """
        返回：
            newly_cleaned, revisited_count
        """
        radius = self.cfg.robot.cleaning_radius
        rr = radius * radius
        new_clean = 0
        revisited = 0

        min_gx = max(0, int((self.robot_x - radius) // self.grid))
        max_gx = min(self.grid_w - 1,
                     int((self.robot_x + radius) // self.grid))
        min_gy = max(0, int((self.robot_y - radius) // self.grid))
        max_gy = min(self.grid_h - 1,
                     int((self.robot_y + radius) // self.grid))

        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                if self.obstacle_map[gy, gx] > 0.5:
                    continue

                cx = gx * self.grid + self.grid * 0.5
                cy = gy * self.grid + self.grid * 0.5
                dx = cx - self.robot_x
                dy = cy - self.robot_y

                if dx * dx + dy * dy <= rr:
                    if self.clean_map[gy, gx] < 0.5:
                        self.clean_map[gy, gx] = 1.0
                        new_clean += 1
                    else:
                        revisited += 1

        return new_clean, revisited

    def _coverage(self) -> float:
        if self.total_cleanable_cells <= 0:
            return 0.0

        cleaned = np.logical_and(self.clean_map > 0.5,
                                 self.obstacle_map < 0.5).sum()
        return float(cleaned / self.total_cleanable_cells)

    def _cast_rays(self) -> np.ndarray:
        """
        返回 shape=(num_rays, 3)
        每行: [distance_norm, hit_static, hit_dynamic]
        """
        num_rays = self.cfg.sensor.num_rays
        max_dist = self.cfg.sensor.ray_max_distance
        fov = math.radians(self.cfg.sensor.ray_fov_deg)
        step = self.cfg.sensor.ray_step

        hits = np.zeros((num_rays, 3), dtype=np.float32)

        if num_rays == 1:
            angles = [self.robot_theta]
        else:
            angles = np.linspace(
                self.robot_theta - fov / 2.0,
                self.robot_theta + fov / 2.0,
                num_rays
            )

        for i, angle in enumerate(angles):
            dist = max_dist
            hit_static = 0.0
            hit_dynamic = 0.0

            t = 0.0
            while t <= max_dist:
                px = self.robot_x + t * math.cos(angle)
                py = self.robot_y + t * math.sin(angle)

                if self._point_hits_static(px, py):
                    dist = t
                    hit_static = 1.0
                    break
                if self._point_hits_dynamic(px, py):
                    dist = t
                    hit_dynamic = 1.0
                    break

                t += step

            hits[i, 0] = float(np.clip(dist / max_dist, 0.0, 1.0))
            hits[i, 1] = hit_static
            hits[i, 2] = hit_dynamic

        self.last_ray_hits = hits
        self.last_ray_summary = self._summarize_rays(hits)
        return hits

    def _summarize_rays(self, rays: np.ndarray) -> Dict[str, float]:
        """
        将原始 rays 额外压成几个摘要量，降低网络提特征难度。
        """
        num_rays = len(rays)
        center_idx = num_rays // 2
        half_span = self.cfg.sensor.front_ray_half_span

        front_l = max(0, center_idx - half_span)
        front_r = min(num_rays, center_idx + half_span + 1)

        front_slice = rays[front_l:front_r]
        left_slice = rays[:center_idx] if center_idx > 0 else rays[:1]
        right_slice = rays[center_idx + 1:] if center_idx + \
            1 < num_rays else rays[-1:]

        front_min = float(np.min(front_slice[:, 0])) if len(
            front_slice) > 0 else 1.0
        min_dist = float(np.min(rays[:, 0])) if len(rays) > 0 else 1.0
        left_mean = float(np.mean(left_slice[:, 0])) if len(
            left_slice) > 0 else 1.0
        right_mean = float(np.mean(right_slice[:, 0])) if len(
            right_slice) > 0 else 1.0

        dynamic_close_mask = np.logical_and(
            front_slice[:, 2] > 0.5,
            front_slice[:, 0] <= self.cfg.sensor.dynamic_hit_distance_ratio
        )
        front_has_dynamic = 1.0 if np.any(dynamic_close_mask) else 0.0

        return {
            "front_min_ray_dist": front_min,
            "min_ray_dist": min_dist,
            "left_mean_ray_dist": left_mean,
            "right_mean_ray_dist": right_mean,
            "front_has_dynamic": front_has_dynamic,
        }

    def _extract_local_patch(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        从全局网格裁切机器人周围局部 patch
        """
        L = self.cfg.sensor.local_map_size
        half = L // 2

        gx = int(self.robot_x // self.grid)
        gy = int(self.robot_y // self.grid)

        clean_patch = np.zeros((L, L), dtype=np.float32)
        obstacle_patch = np.ones((L, L), dtype=np.float32)
        dock_patch = np.zeros(
            (L, L), dtype=np.float32) if self.cfg.sensor.use_local_dock_map else None

        for py in range(L):
            for px in range(L):
                src_x = gx + (px - half)
                src_y = gy + (py - half)

                if 0 <= src_x < self.grid_w and 0 <= src_y < self.grid_h:
                    clean_patch[py, px] = self.clean_map[src_y, src_x]
                    obstacle_patch[py, px] = self.obstacle_map[src_y, src_x]
                    if dock_patch is not None:
                        dock_patch[py, px] = self.dock_map[src_y, src_x]
                else:
                    clean_patch[py, px] = 0.0
                    obstacle_patch[py, px] = 1.0
                    if dock_patch is not None:
                        dock_patch[py, px] = 0.0

        for obs in self.dynamic_obstacles:
            ogx = int(obs.x // self.grid)
            ogy = int(obs.y // self.grid)

            px = ogx - gx + half
            py = ogy - gy + half
            if 0 <= px < L and 0 <= py < L:
                obstacle_patch[py, px] = 1.0

        return clean_patch, obstacle_patch, dock_patch

    def _battery_can_return_hint(self, dist_to_dock: float) -> float:
        """
        粗略回桩可行性信号：
        按“到桩所需步数”与“剩余电量”估一个 0~1 提示。
        该值只做 hint，不追求精确物理。
        """
        avg_speed = max(self.cfg.robot.max_linear_speed * 0.55, 1e-6)
        est_steps = dist_to_dock / (avg_speed * self.dt)
        est_energy_need = est_steps * (
            self.cfg.robot.battery_per_step + self.cfg.robot.battery_move_scale * 0.55
        )
        margin = self.battery - est_energy_need
        hint = margin / max(self._stage_battery_capacity * 0.25, 1e-6)
        return float(np.clip(hint, 0.0, 1.0))

    def _get_state_vector(self) -> np.ndarray:
        battery_ratio = self.battery / max(self._stage_battery_capacity, 1e-6)
        coverage = self._coverage()

        dock_dx, dock_dy = self._dock_direction(self.robot_x, self.robot_y)
        dock_dx_norm = np.clip(dock_dx / max(self.W, 1), -1.0, 1.0)
        dock_dy_norm = np.clip(dock_dy / max(self.H, 1), -1.0, 1.0)

        dist_to_dock = math.hypot(dock_dx, dock_dy)
        dist_to_dock_norm = float(
            np.clip(dist_to_dock / max(math.hypot(self.W, self.H), 1e-6), 0.0, 1.0))

        target_theta = math.atan2(dock_dy, dock_dx)
        dock_angle_error = self._normalize_angle(
            target_theta - self.robot_theta) / math.pi
        dock_angle_error = float(np.clip(dock_angle_error, -1.0, 1.0))

        is_low_battery = 1.0 if battery_ratio <= self.cfg.robot.low_battery_threshold else 0.0
        on_dock = 1.0 if self._robot_on_dock() else 0.0

        ray_summary = self.last_ray_summary if self.last_ray_summary else {
            "front_min_ray_dist": 1.0,
            "min_ray_dist": 1.0,
            "left_mean_ray_dist": 1.0,
            "right_mean_ray_dist": 1.0,
            "front_has_dynamic": 0.0,
        }

        battery_can_return_hint = self._battery_can_return_hint(dist_to_dock)

        state = np.array(
            [
                self.robot_x / self.W,
                self.robot_y / self.H,
                math.cos(self.robot_theta),
                math.sin(self.robot_theta),
                self.robot_v / max(self.cfg.robot.max_linear_speed, 1e-6),
                self.robot_w / max(self.cfg.robot.max_angular_speed, 1e-6),
                battery_ratio,
                coverage,
                dock_dx_norm,
                dock_dy_norm,
                is_low_battery,
                on_dock,
                dist_to_dock_norm,
                dock_angle_error,
                ray_summary["front_min_ray_dist"],
                ray_summary["min_ray_dist"],
                ray_summary["left_mean_ray_dist"],
                ray_summary["right_mean_ray_dist"],
                ray_summary["front_has_dynamic"],
                battery_can_return_hint,
            ],
            dtype=np.float32,
        )
        return state

    def _get_obs(self):
        rays = self._cast_rays()
        state = self._get_state_vector()
        local_clean_map, local_obstacle_map, local_dock_map = self._extract_local_patch()

        if self.cfg.use_dict_observation:
            obs = {
                "state": state,
                "rays": rays.astype(np.float32),
                "local_clean_map": local_clean_map.astype(np.float32),
                "local_obstacle_map": local_obstacle_map.astype(np.float32),
            }
            if self.cfg.sensor.use_local_dock_map and local_dock_map is not None:
                obs["local_dock_map"] = local_dock_map.astype(np.float32)
            return obs

        chunks = [
            state.flatten(),
            rays.flatten(),
            local_clean_map.flatten(),
            local_obstacle_map.flatten(),
        ]
        if self.cfg.sensor.use_local_dock_map and local_dock_map is not None:
            chunks.append(local_dock_map.flatten())

        flat = np.concatenate(chunks).astype(np.float32)
        return np.clip(flat, -1.0, 1.0)

    # =========================================================
    # Gym 接口
    # =========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._np_random is None or seed is not None:
            self._seed(seed if seed is not None else self.cfg.seed)

        self.steps = 0
        self.clean_map.fill(0.0)
        self.last_done_reason = None

        self.idle_steps = 0
        self.no_progress_steps = 0
        self.dock_idle_steps = 0

        self._build_dock()
        self._build_valid_map()
        self._spawn_dynamic_obstacles()

        self.robot_x, self.robot_y, self.robot_theta = self._random_free_pose()
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.battery = self._stage_battery_capacity

        self._mark_cleaned()

        _ = self._cast_rays()

        self.prev_coverage = self._coverage()
        self.prev_dist_to_dock = self._dist_to_dock(self.robot_x, self.robot_y)
        self.prev_on_dock = self._robot_on_dock()
        self.low_battery_entered = False

        obs = self._get_obs()
        info = {
            "coverage": self.prev_coverage,
            "battery": self.battery,
            "battery_ratio": self.battery / max(self._stage_battery_capacity, 1e-6),
            "steps": self.steps,
            "cleanable_cells": self.total_cleanable_cells,
            "dynamic_obstacles": len(self.dynamic_obstacles),
            "training_mode": self.cfg.training_mode,
            "idle_steps": self.idle_steps,
            "no_progress_steps": self.no_progress_steps,
            "dock_idle_steps": self.dock_idle_steps,
            "done_reason": None,
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.steps += 1
        reward_raw = 0.0
        terminated = False
        truncated = False
        done_reason = None

        prev_x = self.robot_x
        prev_y = self.robot_y

        # 先移动动态障碍
        self._move_dynamic_obstacles()

        # 动作映射
        self.robot_v = float(action[0]) * self.cfg.robot.max_linear_speed
        self.robot_w = float(action[1]) * self.cfg.robot.max_angular_speed

        new_theta = self._normalize_angle(
            self.robot_theta + self.robot_w * self.dt)
        new_x = self.robot_x + self.robot_v * math.cos(new_theta) * self.dt
        new_y = self.robot_y + self.robot_v * math.sin(new_theta) * self.dt

        # 碰撞检测
        coll_static, coll_dynamic = self._robot_collides_any(new_x, new_y)

        if coll_static or coll_dynamic:
            if coll_static:
                reward_raw += self.cfg.reward.collision_static_penalty
                done_reason = "collision_static"
            if coll_dynamic:
                reward_raw += self.cfg.reward.collision_dynamic_penalty
                done_reason = "collision_dynamic" if done_reason is None else done_reason

            terminated = True
            self.robot_v = 0.0
        else:
            self.robot_x = new_x
            self.robot_y = new_y
            self.robot_theta = new_theta

        actual_move_dist = math.hypot(
            self.robot_x - prev_x, self.robot_y - prev_y)

        # 电量变化
        move_cost = abs(self.robot_v) / \
            max(self.cfg.robot.max_linear_speed, 1e-6)
        battery_cost = self.cfg.robot.battery_per_step + \
            self.cfg.robot.battery_move_scale * move_cost
        self.battery -= battery_cost

        # 在充电桩内回充
        on_dock = self._robot_on_dock()
        recharge_amount = 0.0
        if on_dock:
            recharge_amount = self.cfg.robot.dock_recharge_rate
            self.battery += recharge_amount

        self.battery = float(
            np.clip(self.battery, 0.0, self._stage_battery_capacity))
        battery_ratio = self.battery / max(self._stage_battery_capacity, 1e-6)

        # 清扫
        newly_cleaned, revisited = self._mark_cleaned()
        coverage = self._coverage()
        coverage_gain = coverage - self.prev_coverage

        # 重新获取 ray 摘要
        _ = self._cast_rays()
        ray_summary = self.last_ray_summary
        min_ray_dist = ray_summary["min_ray_dist"]
        front_min_ray_dist = ray_summary["front_min_ray_dist"]

        # 距 dock 变化
        dist_to_dock = self._dist_to_dock(self.robot_x, self.robot_y)
        dist_progress = self.prev_dist_to_dock - dist_to_dock

        # -----------------------------------------------------
        # 基础 reward
        # -----------------------------------------------------
        reward_raw += self.cfg.reward.step_penalty

        low_battery = battery_ratio <= self.cfg.robot.low_battery_threshold
        if low_battery:
            self.low_battery_entered = True

        # -----------------------------------------------------
        # 主任务 reward
        # -----------------------------------------------------
        if not low_battery:
            reward_raw += newly_cleaned * self.cfg.reward.new_clean_cell_reward
            reward_raw += coverage_gain * self.cfg.reward.coverage_gain_reward
        else:
            reward_raw += (
                dist_progress
                * self.cfg.reward.low_battery_to_dock_progress_reward
                / max(self.W, self.H)
            )
            if dist_progress < 0:
                reward_raw += self.cfg.reward.low_battery_away_from_dock_penalty

            reward_raw += newly_cleaned * (
                self.cfg.reward.new_clean_cell_reward *
                self.cfg.reward.low_battery_clean_reward_scale
            )
            reward_raw += coverage_gain * (
                self.cfg.reward.coverage_gain_reward *
                self.cfg.reward.low_battery_clean_reward_scale
            )

        # -----------------------------------------------------
        # 充电桩 reward / penalty
        # -----------------------------------------------------
        if on_dock:
            if low_battery:
                reward_raw += self.cfg.reward.dock_contact_reward
                reward_raw += recharge_amount * self.cfg.reward.recharge_reward_scale
            else:
                reward_raw += self.cfg.reward.non_low_battery_dock_penalty

                if battery_ratio >= self.cfg.reward.high_battery_threshold_for_dock_penalty:
                    reward_raw += self.cfg.reward.high_battery_dock_penalty

                if coverage <= self.cfg.reward.early_stage_coverage_threshold:
                    reward_raw += self.cfg.reward.early_stage_dock_penalty

        if self.low_battery_entered and on_dock and (not self.prev_on_dock):
            reward_raw += self.cfg.reward.low_battery_return_success_bonus

        # -----------------------------------------------------
        # 反挂机计数器
        # -----------------------------------------------------
        idle_linear_speed = abs(self.robot_v) / \
            max(self.cfg.robot.max_linear_speed, 1e-6)
        idle_angular_speed = abs(self.robot_w) / \
            max(self.cfg.robot.max_angular_speed, 1e-6)

        is_idle = (
            idle_linear_speed <= self.cfg.reward.idle_linear_speed_ratio_threshold
            and idle_angular_speed <= self.cfg.reward.idle_angular_speed_ratio_threshold
            and actual_move_dist <= self.cfg.robot.radius * 0.05
        )

        has_progress = (newly_cleaned > 0) or (coverage_gain > 1e-8)

        if is_idle:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        if has_progress:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        if on_dock and (not low_battery) and is_idle:
            self.dock_idle_steps += 1
        else:
            self.dock_idle_steps = 0

        # -----------------------------------------------------
        # 反挂机惩罚
        # -----------------------------------------------------
        if self.idle_steps >= self.cfg.reward.idle_step_threshold:
            reward_raw += self.cfg.reward.idle_penalty

        if self.no_progress_steps >= self.cfg.reward.no_progress_step_threshold:
            reward_raw += self.cfg.reward.no_progress_penalty

        if self.dock_idle_steps >= self.cfg.reward.dock_idle_step_threshold:
            reward_raw += self.cfg.reward.dock_idle_penalty

        # -----------------------------------------------------
        # 结束条件
        # -----------------------------------------------------
        if coverage >= 0.995 and not terminated:
            reward_raw += self.cfg.reward.full_clean_bonus
            terminated = True
            done_reason = "full_clean"

        if self.battery <= 0.0 and not terminated:
            reward_raw += self.cfg.reward.battery_empty_penalty
            terminated = True
            done_reason = "battery_empty"

        if self.steps >= self.max_steps and not terminated:
            reward_raw += self.cfg.reward.timeout_penalty
            truncated = True
            done_reason = "timeout"

        self.prev_coverage = coverage
        self.prev_dist_to_dock = dist_to_dock
        self.prev_on_dock = on_dock
        self.last_done_reason = done_reason

        reward = float(reward_raw * self.cfg.reward.reward_scale)

        obs = self._get_obs()
        info = {
            "coverage": coverage,
            "coverage_gain": coverage_gain,
            "battery": self.battery,
            "battery_ratio": battery_ratio,
            "steps": self.steps,
            "collision_static": coll_static,
            "collision_dynamic": coll_dynamic,
            "newly_cleaned": newly_cleaned,
            "revisited": revisited,
            "on_dock": on_dock,
            "dist_to_dock": dist_to_dock,
            "dist_to_dock_norm": float(np.clip(dist_to_dock / max(math.hypot(self.W, self.H), 1e-6), 0.0, 1.0)),
            "front_min_ray_dist": front_min_ray_dist,
            "min_ray_dist": min_ray_dist,
            "left_mean_ray_dist": ray_summary["left_mean_ray_dist"],
            "right_mean_ray_dist": ray_summary["right_mean_ray_dist"],
            "front_has_dynamic": ray_summary["front_has_dynamic"],
            "battery_can_return_hint": self._battery_can_return_hint(dist_to_dock),
            "idle_steps": self.idle_steps,
            "no_progress_steps": self.no_progress_steps,
            "dock_idle_steps": self.dock_idle_steps,
            "reward_raw": float(reward_raw),
            "reward": reward,
            "done_reason": done_reason,
            "training_mode": self.cfg.training_mode,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # =========================================================
    # render
    # =========================================================
    def _init_pygame(self) -> None:
        if self.window is None:
            pygame.init()

            if self.render_mode == "human":
                self.window = pygame.display.set_mode((self.W, self.H))
                pygame.display.set_caption(self.cfg.render.window_caption)

            self.clock = pygame.time.Clock()
            self.canvas = pygame.Surface((self.W, self.H))

    def render(self):
        if self.render_mode is None:
            return None

        self._init_pygame()
        self.canvas.fill((238, 238, 238))

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                x = gx * self.grid
                y = gy * self.grid
                rect = pygame.Rect(x, y, self.grid, self.grid)

                if self.obstacle_map[gy, gx] > 0.5:
                    continue

                if self.clean_map[gy, gx] > 0.5:
                    pygame.draw.rect(self.canvas, (205, 240, 205), rect)
                else:
                    pygame.draw.rect(self.canvas, (245, 245, 245), rect)

                if self.dock_map[gy, gx] > 0.5:
                    pygame.draw.rect(self.canvas, (170, 210, 255), rect)

                if self.cfg.render.draw_grid:
                    pygame.draw.rect(self.canvas, (224, 224, 224), rect, 1)

        bt = self.cfg.map_cfg.border_thickness
        pygame.draw.rect(self.canvas, (50, 50, 50),
                         pygame.Rect(0, 0, self.W, bt))
        pygame.draw.rect(self.canvas, (50, 50, 50),
                         pygame.Rect(0, self.H - bt, self.W, bt))
        pygame.draw.rect(self.canvas, (50, 50, 50),
                         pygame.Rect(0, 0, bt, self.H))
        pygame.draw.rect(self.canvas, (50, 50, 50),
                         pygame.Rect(self.W - bt, 0, bt, self.H))

        for rect in self.obstacles:
            pygame.draw.rect(self.canvas, (95, 95, 95), rect)

        pygame.draw.rect(self.canvas, (80, 150, 255), self.dock_rect)

        for obs in self.dynamic_obstacles:
            pygame.draw.circle(
                self.canvas,
                (255, 165, 60),
                (int(obs.x), int(obs.y)),
                int(obs.radius),
            )

        robot_pos = (int(self.robot_x), int(self.robot_y))
        pygame.draw.circle(
            self.canvas,
            (220, 80, 80),
            robot_pos,
            int(self.cfg.robot.radius),
        )

        hx = self.robot_x + self.cfg.robot.radius * math.cos(self.robot_theta)
        hy = self.robot_y + self.cfg.robot.radius * math.sin(self.robot_theta)
        pygame.draw.line(self.canvas, (255, 255, 255),
                         robot_pos, (int(hx), int(hy)), 3)

        if self.cfg.render.draw_rays and self.last_ray_hits is not None:
            num_rays = self.cfg.sensor.num_rays
            max_dist = self.cfg.sensor.ray_max_distance
            fov = math.radians(self.cfg.sensor.ray_fov_deg)

            if num_rays == 1:
                angles = [self.robot_theta]
            else:
                angles = np.linspace(
                    self.robot_theta - fov / 2.0,
                    self.robot_theta + fov / 2.0,
                    num_rays
                )

            for i, angle in enumerate(angles):
                dist = self.last_ray_hits[i, 0] * max_dist
                hit_static = self.last_ray_hits[i, 1] > 0.5
                hit_dynamic = self.last_ray_hits[i, 2] > 0.5

                px = self.robot_x + dist * math.cos(angle)
                py = self.robot_y + dist * math.sin(angle)

                if hit_dynamic:
                    color = (255, 180, 60)
                elif hit_static:
                    color = (120, 170, 255)
                else:
                    color = (200, 200, 200)

                pygame.draw.line(
                    self.canvas,
                    color,
                    (int(self.robot_x), int(self.robot_y)),
                    (int(px), int(py)),
                    1
                )

        if self.cfg.render.draw_status_text:
            coverage = self._coverage()
            battery_ratio = self.battery / \
                max(self._stage_battery_capacity, 1e-6)
            summary = self.last_ray_summary if self.last_ray_summary else {}
            text = (
                f"step={self.steps}   "
                f"coverage={coverage:.3f}   "
                f"battery={self.battery:.1f} ({battery_ratio:.2f})   "
                f"front={summary.get('front_min_ray_dist', 1.0):.2f}   "
                f"dyn={len(self.dynamic_obstacles)}   "
                f"idle={self.idle_steps}   "
                f"noprog={self.no_progress_steps}   "
                f"mode={self.cfg.training_mode}"
            )
            font = pygame.font.SysFont("consolas", 18)
            text_surf = font.render(text, True, (20, 20, 20))
            self.canvas.blit(text_surf, (10, 10))

            if self.last_done_reason is not None:
                text2 = f"done_reason={self.last_done_reason}"
                text2_surf = font.render(text2, True, (20, 20, 20))
                self.canvas.blit(text2_surf, (10, 34))

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            self.window.blit(self.canvas, (0, 0))
            pygame.display.update()
            self.clock.tick(self.cfg.render.render_fps)
            return None

        if self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(self.canvas)
            return np.transpose(arr, (1, 0, 2))

        return None

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
            self.canvas = None

    # =========================================================
    # demo policy
    # =========================================================
    def heuristic_policy(self) -> np.ndarray:
        """
        一个简单的手写策略：
        - 电量低时优先回充
        - 否则朝最近未清扫区域走
        - 若前方很近有障碍，优先转向避让
        """
        rays = self._cast_rays()
        front_dist = self.last_ray_summary["front_min_ray_dist"]
        battery_ratio = self.battery / max(self._stage_battery_capacity, 1e-6)

        if front_dist < 0.18:
            left_score = self.last_ray_summary["left_mean_ray_dist"]
            right_score = self.last_ray_summary["right_mean_ray_dist"]
            if left_score > right_score:
                return np.array([0.15, -0.9], dtype=np.float32)
            return np.array([0.15, 0.9], dtype=np.float32)

        if battery_ratio <= self.cfg.robot.low_battery_threshold:
            tx = self.dock_rect.centerx
            ty = self.dock_rect.centery
        else:
            target = None
            best_dist = 1e18
            for gy in range(self.grid_h):
                for gx in range(self.grid_w):
                    if self.obstacle_map[gy, gx] > 0.5:
                        continue
                    if self.clean_map[gy, gx] > 0.5:
                        continue
                    if self.reachable_from_dock_map[gy, gx] < 0.5:
                        continue

                    cx = gx * self.grid + self.grid * 0.5
                    cy = gy * self.grid + self.grid * 0.5
                    d = (cx - self.robot_x) ** 2 + (cy - self.robot_y) ** 2
                    if d < best_dist:
                        best_dist = d
                        target = (cx, cy)

            if target is None:
                tx = self.dock_rect.centerx
                ty = self.dock_rect.centery
            else:
                tx, ty = target

        dx = tx - self.robot_x
        dy = ty - self.robot_y
        target_theta = math.atan2(dy, dx)
        angle_error = self._normalize_angle(target_theta - self.robot_theta)

        w = np.clip(angle_error / math.pi, -1.0, 1.0)

        v = 0.95 * max(0.0, 1.0 - abs(w))
        v *= np.clip(front_dist / 0.4, 0.2, 1.0)

        if self._robot_on_dock() and battery_ratio < 0.95:
            return np.array([0.0, 0.0], dtype=np.float32)

        return np.array([v, w], dtype=np.float32)


# =========================================================
# 测试接口
# =========================================================
def run_random_demo(steps: int = 500, seed: int = 42) -> None:
    print("=" * 70)
    print("Random Demo")
    print("=" * 70)

    cfg = EnvConfig()
    env = CleaningRobotEnv(cfg=cfg, render_mode="human")

    obs, info = env.reset(seed=seed)
    print("reset info:", info)

    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if info["steps"] % 50 == 0:
            print(
                f"[Random] step={info['steps']}, "
                f"reward={reward:.3f}, "
                f"coverage={info['coverage']:.3f}, "
                f"battery={info['battery']:.1f}, "
                f"on_dock={info['on_dock']}, "
                f"idle={info['idle_steps']}, "
                f"noprog={info['no_progress_steps']}, "
                f"dock_idle={info['dock_idle_steps']}, "
                f"done_reason={info['done_reason']}, "
                f"coll_s={info['collision_static']}, "
                f"coll_d={info['collision_dynamic']}"
            )

        if terminated or truncated:
            break

    print("episode end:", info)
    env.close()


def run_heuristic_demo(steps: int = 1600, seed: int = 42) -> None:
    print("=" * 70)
    print("Heuristic Demo")
    print("=" * 70)

    cfg = EnvConfig()
    env = CleaningRobotEnv(cfg=cfg, render_mode="human")

    obs, info = env.reset(seed=seed)
    print("reset info:", info)

    for _ in range(steps):
        action = env.heuristic_policy()
        obs, reward, terminated, truncated, info = env.step(action)

        if info["steps"] % 50 == 0:
            print(
                f"[Heuristic] step={info['steps']}, "
                f"reward={reward:.3f}, "
                f"coverage={info['coverage']:.3f}, "
                f"battery={info['battery']:.1f}, "
                f"on_dock={info['on_dock']}, "
                f"new_clean={info['newly_cleaned']}, "
                f"idle={info['idle_steps']}, "
                f"noprog={info['no_progress_steps']}, "
                f"dock_idle={info['dock_idle_steps']}, "
                f"done_reason={info['done_reason']}, "
                f"coll_s={info['collision_static']}, "
                f"coll_d={info['collision_dynamic']}"
            )

        if terminated or truncated:
            break

    print("episode end:", info)
    env.close()


if __name__ == "__main__":
    """
    运行方式：
        python cleaning_robot_env.py

    依赖：
        pip install gymnasium pygame numpy
    """
    run_random_demo(steps=400, seed=42)
    run_heuristic_demo(steps=1600, seed=42)
