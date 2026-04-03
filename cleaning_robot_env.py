import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from env_cfg import EnvConfig


@dataclass
class RectObstacle:
    rect: pygame.Rect
    kind: str  # "block" or "wall"


@dataclass
class DynamicObstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float


class CleaningRobotEnv(gym.Env):
    """
    精简版二维连续扫地机器人环境

    难度：
    - easy   : 无障碍物
    - medium : 静态障碍物
    - hard   : 静态 + 动态障碍物

    规则：
    - 初始从 dock 出发，满电
    - 动作是线加速度 + 角加速度
    - 回到 dock 后必须充满电，充满后自动重置到 dock 中心
    - 不再因 coverage 达标提前结束
    - 撞静态障碍、动态障碍、边界：结束
    - 超时：结束
    - 射线可看到障碍物和未清扫污渍
    - 污渍不是全图，而是随机生成的聚落
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvConfig()
        self.render_mode = render_mode
        self._render_enabled = render_mode is not None

        self.W = self.cfg.map_cfg.width
        self.H = self.cfg.map_cfg.height
        self.grid = self.cfg.map_cfg.clean_grid_size
        self.grid_w = self.W // self.grid
        self.grid_h = self.H // self.grid
        self.dt = self.cfg.dt
        self.max_steps = self.cfg.max_steps

        self._np_random: Optional[np.random.Generator] = None

        xs = np.arange(self.grid_w, dtype=np.float32) * \
            self.grid + self.grid * 0.5
        ys = np.arange(self.grid_h, dtype=np.float32) * \
            self.grid + self.grid * 0.5
        self.cell_center_x, self.cell_center_y = np.meshgrid(xs, ys)
        self.grid_x_idx, self.grid_y_idx = np.meshgrid(
            np.arange(self.grid_w, dtype=np.int32),
            np.arange(self.grid_h, dtype=np.int32),
        )

        # 动作：线加速度 + 角加速度
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # state:
        # [x/W, y/H, cos(theta), sin(theta),
        #  v/vmax, w/wmax,
        #  battery_ratio,
        #  on_dock, charging_locked,
        #  dock_dx_norm, dock_dy_norm, dock_dist_norm,
        #  front_obstacle_dist, min_obstacle_dist,
        #  front_unclean_dist, min_unclean_dist]
        self.state_dim = 16

        local_size = self.cfg.sensor.local_map_size
        ray_shape = (self.cfg.sensor.num_rays, 5)
        # [obs_dist, hit_static, hit_dynamic, unclean_dist, hit_unclean]

        if self.cfg.use_dict_observation:
            self.observation_space = spaces.Dict({
                "state": spaces.Box(low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32),
                "rays": spaces.Box(low=0.0, high=1.0, shape=ray_shape, dtype=np.float32),
                "local_clean_map": spaces.Box(low=0.0, high=1.0, shape=(local_size, local_size), dtype=np.float32),
                "local_obstacle_map": spaces.Box(low=0.0, high=1.0, shape=(local_size, local_size), dtype=np.float32),
            })
        else:
            flat_dim = self.state_dim + self.cfg.sensor.num_rays * \
                5 + local_size * local_size * 2
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(flat_dim,), dtype=np.float32
            )

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.battery = 0.0

        self.steps = 0

        # clean_map: 已清扫污渍
        # dirt_map : 初始污渍区域
        self.clean_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.dirt_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.visit_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.obstacle_map = np.zeros(
            (self.grid_h, self.grid_w), dtype=np.float32)

        self.total_cleanable_cells = 0
        self.total_dirty_cells = 0

        self.obstacles: List[RectObstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []
        self.dock_rect: Optional[pygame.Rect] = None

        self.last_done_reason: Optional[str] = None
        self.last_ray_hits: Optional[np.ndarray] = None
        self.last_ray_summary: Dict[str, float] = {}

        self.charging_locked = False
        self.prev_seen_unclean = False

        self.window = None
        self.clock = None
        self.canvas = None

    # =========================================================
    # 工具
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

    def _circle_circle_collision(
        self,
        x1: float,
        y1: float,
        r1: float,
        x2: float,
        y2: float,
        r2: float,
    ) -> bool:
        dx = x1 - x2
        dy = y1 - y2
        return dx * dx + dy * dy <= (r1 + r2) * (r1 + r2)

    def _build_dock(self) -> None:
        s = self.cfg.map_cfg.dock_size
        x, y = self.cfg.map_cfg.dock_pos
        self.dock_rect = pygame.Rect(x, y, s, s)

    def _reset_robot_to_dock_center(self) -> None:
        self.robot_x = float(self.dock_rect.centerx)
        self.robot_y = float(self.dock_rect.centery)
        self.robot_theta = float(self._np_random.uniform(-math.pi, math.pi))
        self.robot_v = 0.0
        self.robot_w = 0.0

    def _dist_to_dock(self, x: float, y: float) -> float:
        cx = self.dock_rect.centerx
        cy = self.dock_rect.centery
        return math.hypot(x - cx, y - cy)

    def _dock_direction(self, x: float, y: float) -> Tuple[float, float]:
        cx = self.dock_rect.centerx
        cy = self.dock_rect.centery
        return cx - x, cy - y

    def _robot_on_dock(self, x: Optional[float] = None, y: Optional[float] = None) -> bool:
        if x is None:
            x = self.robot_x
        if y is None:
            y = self.robot_y
        return self._circle_rect_collision(x, y, self.cfg.robot.radius, self.dock_rect)

    def _point_hits_static(self, x: float, y: float) -> bool:
        bt = self.cfg.map_cfg.border_thickness
        if x < bt or x > self.W - bt or y < bt or y > self.H - bt:
            return True
        for obs in self.obstacles:
            if obs.rect.collidepoint(x, y):
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
        if x - r < bt or x + r > self.W - bt or y - r < bt or y + r > self.H - bt:
            return True
        for obs in self.obstacles:
            if self._circle_rect_collision(x, y, r, obs.rect):
                return True
        return False

    def _robot_collides_dynamic(self, x: float, y: float) -> bool:
        r = self.cfg.robot.radius
        for obs in self.dynamic_obstacles:
            if self._circle_circle_collision(x, y, r, obs.x, obs.y, obs.radius):
                return True
        return False

    def _difficulty(self) -> str:
        mode = str(self.cfg.difficulty).lower()
        if mode not in {"easy", "medium", "hard"}:
            return "medium"
        return mode

    # =========================================================
    # 地图 / 障碍
    # =========================================================
    def _generate_random_obstacles(self) -> None:
        self.obstacles = []

        if self._difficulty() == "easy":
            return

        mcfg = self.cfg.map_cfg
        dock_avoid = self.dock_rect.inflate(
            mcfg.dock_safe_clearance * 2, mcfg.dock_safe_clearance * 2
        )

        def valid_rect(candidate: pygame.Rect) -> bool:
            bt = mcfg.border_thickness
            if candidate.left < bt + 8 or candidate.right > self.W - bt - 8:
                return False
            if candidate.top < bt + 8 or candidate.bottom > self.H - bt - 8:
                return False
            if candidate.colliderect(dock_avoid):
                return False
            for obs in self.obstacles:
                if candidate.inflate(mcfg.obstacle_margin, mcfg.obstacle_margin).colliderect(
                    obs.rect.inflate(mcfg.obstacle_margin,
                                     mcfg.obstacle_margin)
                ):
                    return False
            return True

        b_lo, b_hi = mcfg.block_obstacle_count_range
        block_n = int(self._np_random.integers(b_lo, b_hi + 1))
        for _ in range(block_n):
            for _ in range(200):
                w = int(self._np_random.integers(
                    mcfg.block_min_size[0], mcfg.block_max_size[0] + 1))
                h = int(self._np_random.integers(
                    mcfg.block_min_size[1], mcfg.block_max_size[1] + 1))
                x = int(self._np_random.integers(20, max(21, self.W - w - 20)))
                y = int(self._np_random.integers(20, max(21, self.H - h - 20)))
                rect = pygame.Rect(x, y, w, h)
                if valid_rect(rect):
                    self.obstacles.append(
                        RectObstacle(rect=rect, kind="block"))
                    break

        w_lo, w_hi = mcfg.wall_obstacle_count_range
        wall_n = int(self._np_random.integers(w_lo, w_hi + 1))
        for _ in range(wall_n):
            for _ in range(200):
                thickness = int(self._np_random.integers(
                    mcfg.wall_thickness_range[0], mcfg.wall_thickness_range[1] + 1
                ))
                length = int(self._np_random.integers(
                    mcfg.wall_length_range[0], mcfg.wall_length_range[1] + 1
                ))
                horizontal = bool(self._np_random.integers(0, 2))

                if horizontal:
                    rect = pygame.Rect(
                        int(self._np_random.integers(
                            20, max(21, self.W - length - 20))),
                        int(self._np_random.integers(
                            20, max(21, self.H - thickness - 20))),
                        length,
                        thickness,
                    )
                else:
                    rect = pygame.Rect(
                        int(self._np_random.integers(
                            20, max(21, self.W - thickness - 20))),
                        int(self._np_random.integers(
                            20, max(21, self.H - length - 20))),
                        thickness,
                        length,
                    )

                if valid_rect(rect):
                    self.obstacles.append(RectObstacle(rect=rect, kind="wall"))
                    break

    def _build_obstacle_map(self) -> None:
        self.obstacle_map.fill(0.0)

        left = self.grid_x_idx * self.grid
        top = self.grid_y_idx * self.grid
        right = left + self.grid
        bottom = top + self.grid

        for obs in self.obstacles:
            rect = obs.rect
            overlap = (
                (right > rect.left) & (left < rect.right) &
                (bottom > rect.top) & (top < rect.bottom)
            )
            self.obstacle_map[overlap] = 1.0

        dl = self.dock_rect.left
        dt = self.dock_rect.top
        dr = self.dock_rect.right
        db = self.dock_rect.bottom
        dock_overlap = (
            (right > dl) & (left < dr) &
            (bottom > dt) & (top < db)
        )
        self.obstacle_map[dock_overlap] = 0.0

        self.total_cleanable_cells = int((self.obstacle_map < 0.5).sum())

    def _map_is_valid(self) -> bool:
        free_ratio = float((self.obstacle_map < 0.5).sum()) / \
            float(self.grid_h * self.grid_w)
        return free_ratio >= self.cfg.map_cfg.min_cleanable_ratio

    def _build_valid_map(self, max_retry: int = 60) -> None:
        for _ in range(max_retry):
            self._generate_random_obstacles()
            self._build_obstacle_map()
            if self._map_is_valid():
                return
        self._build_obstacle_map()

    # =========================================================
    # 污渍聚落
    # =========================================================
    def _free_cell_mask(self) -> np.ndarray:
        free_mask = (self.obstacle_map < 0.5).astype(np.float32)

        left = self.grid_x_idx * self.grid
        top = self.grid_y_idx * self.grid
        right = left + self.grid
        bottom = top + self.grid

        dl = self.dock_rect.left
        dt = self.dock_rect.top
        dr = self.dock_rect.right
        db = self.dock_rect.bottom
        dock_overlap = (
            (right > dl) & (left < dr) &
            (bottom > dt) & (top < db)
        )

        free_mask[dock_overlap] = 0.0
        return free_mask

    def _build_dirt_map(self) -> None:
        self.dirt_map.fill(0.0)

        if not self.cfg.dirt.enabled:
            self.total_dirty_cells = 0
            return

        dcfg = self.cfg.dirt
        free_mask = self._free_cell_mask() > 0.5
        free_indices = np.argwhere(free_mask)

        if len(free_indices) == 0:
            self.total_dirty_cells = 0
            return

        dock_cx = self.dock_rect.centerx
        dock_cy = self.dock_rect.centery

        def valid_cluster_center(gy: int, gx: int) -> bool:
            cx = (gx + 0.5) * self.grid
            cy = (gy + 0.5) * self.grid
            return math.hypot(cx - dock_cx, cy - dock_cy) >= dcfg.spawn_safe_distance_to_dock

        success = False
        for _ in range(dcfg.max_retry):
            self.dirt_map.fill(0.0)

            cluster_count = int(self._np_random.integers(
                dcfg.cluster_count_range[0], dcfg.cluster_count_range[1] + 1
            ))

            valid_centers = [
                (gy, gx) for gy, gx in free_indices
                if valid_cluster_center(int(gy), int(gx))
            ]

            if len(valid_centers) == 0:
                break

            for _ in range(cluster_count):
                center_idx = int(
                    self._np_random.integers(0, len(valid_centers)))
                cy, cx = valid_centers[center_idx]

                radius = int(self._np_random.integers(
                    dcfg.cluster_radius_range[0], dcfg.cluster_radius_range[1] + 1
                ))
                density = float(self._np_random.uniform(
                    dcfg.cluster_density_range[0], dcfg.cluster_density_range[1]
                ))

                y0 = max(0, cy - radius)
                y1 = min(self.grid_h, cy + radius + 1)
                x0 = max(0, cx - radius)
                x1 = min(self.grid_w, cx + radius + 1)

                for gy in range(y0, y1):
                    for gx in range(x0, x1):
                        if not free_mask[gy, gx]:
                            continue

                        dy = gy - cy
                        dx = gx - cx
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist > radius:
                            continue

                        # 中心更密，边缘略稀
                        dist_norm = dist / max(radius, 1e-6)
                        prob = density * (1.0 - 0.55 * dist_norm)

                        # 少量噪声，让形状更自然
                        prob += float(self._np_random.uniform(-0.05, 0.05))
                        prob = float(np.clip(prob, 0.0, 1.0))

                        if self._np_random.random() < prob:
                            self.dirt_map[gy, gx] = 1.0

            self.dirt_map[~free_mask] = 0.0

            dirty_cells = int(self.dirt_map.sum())
            dirty_ratio = dirty_cells / max(int(free_mask.sum()), 1)

            if dcfg.min_dirty_ratio <= dirty_ratio <= dcfg.max_dirty_ratio:
                success = True
                break

        if not success:
            self.dirt_map[~free_mask] = 0.0

        self.total_dirty_cells = int(self.dirt_map.sum())

    # =========================================================
    # 动态障碍
    # =========================================================
    def _spawn_dynamic_obstacles(self) -> None:
        self.dynamic_obstacles = []

        if self._difficulty() != "hard":
            return
        if not self.cfg.dynamic_obstacle.enabled:
            return

        dcfg = self.cfg.dynamic_obstacle
        count = int(self._np_random.integers(
            dcfg.count_range[0], dcfg.count_range[1] + 1))

        for _ in range(count):
            for _ in range(dcfg.max_try_spawn):
                radius = float(self._np_random.integers(
                    dcfg.radius_range[0], dcfg.radius_range[1] + 1))
                speed = float(self._np_random.uniform(
                    dcfg.speed_range[0], dcfg.speed_range[1]))
                angle = float(self._np_random.uniform(-math.pi, math.pi))
                x = float(self._np_random.uniform(50, self.W - 50))
                y = float(self._np_random.uniform(50, self.H - 50))
                vx = speed * math.cos(angle)
                vy = speed * math.sin(angle)

                if self._robot_collides_static(x, y):
                    continue
                if self._dist_to_dock(x, y) < dcfg.spawn_safe_distance_to_dock:
                    continue

                ok = True
                for other in self.dynamic_obstacles:
                    if self._circle_circle_collision(
                        x, y, radius,
                        other.x, other.y, other.radius + dcfg.pair_safe_gap
                    ):
                        ok = False
                        break
                if not ok:
                    continue

                self.dynamic_obstacles.append(
                    DynamicObstacle(x=x, y=y, vx=vx, vy=vy, radius=radius)
                )
                break

    def _move_dynamic_obstacles(self) -> None:
        if not self.dynamic_obstacles:
            return

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
            for rect_obs in self.obstacles:
                if self._circle_rect_collision(nx, ny, obs.radius, rect_obs.rect):
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
    # 清扫 / 探索 / 感知
    # =========================================================
    def _mark_cleaned(self) -> Tuple[int, int]:
        radius = self.cfg.robot.cleaning_radius
        rr = radius * radius
        dx = self.cell_center_x - self.robot_x
        dy = self.cell_center_y - self.robot_y

        mask = (
            (dx * dx + dy * dy <= rr) &
            (self.obstacle_map < 0.5) &
            (self.dirt_map > 0.5)
        )

        newly_cleaned_mask = mask & (self.clean_map < 0.5)
        revisited_mask = mask & (self.clean_map > 0.5)

        new_clean = int(newly_cleaned_mask.sum())
        revisited = int(revisited_mask.sum())

        self.clean_map[newly_cleaned_mask] = 1.0
        return new_clean, revisited

    def _mark_visited(self) -> int:
        gx = int(np.clip(int(self.robot_x // self.grid), 0, self.grid_w - 1))
        gy = int(np.clip(int(self.robot_y // self.grid), 0, self.grid_h - 1))

        if self.obstacle_map[gy, gx] > 0.5:
            return 0

        first_visit = 1 if self.visit_map[gy, gx] < 0.5 else 0
        self.visit_map[gy, gx] = 1.0
        return first_visit

    def _coverage(self) -> float:
        if self.total_dirty_cells <= 0:
            return 0.0
        cleaned = np.logical_and(self.clean_map > 0.5,
                                 self.dirt_map > 0.5).sum()
        return float(cleaned / self.total_dirty_cells)

    def _point_unclean(self, x: float, y: float) -> bool:
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            return False
        gx = int(np.clip(int(x // self.grid), 0, self.grid_w - 1))
        gy = int(np.clip(int(y // self.grid), 0, self.grid_h - 1))
        return (
            (self.obstacle_map[gy, gx] < 0.5) and
            (self.dirt_map[gy, gx] > 0.5) and
            (self.clean_map[gy, gx] < 0.5)
        )

    def _cast_rays(self) -> np.ndarray:
        num_rays = self.cfg.sensor.num_rays
        max_dist = self.cfg.sensor.ray_max_distance
        fov = math.radians(self.cfg.sensor.ray_fov_deg)
        step = self.cfg.sensor.ray_step

        hits = np.zeros((num_rays, 5), dtype=np.float32)

        if num_rays == 1:
            angles = np.array([self.robot_theta], dtype=np.float32)
        else:
            angles = np.linspace(self.robot_theta - fov /
                                 2.0, self.robot_theta + fov / 2.0, num_rays)

        for i, angle in enumerate(angles):
            obs_dist = max_dist
            unclean_dist = max_dist
            hit_static = 0.0
            hit_dynamic = 0.0
            hit_unclean = 0.0
            found_unclean = False

            t = 0.0
            while t <= max_dist:
                px = self.robot_x + t * math.cos(angle)
                py = self.robot_y + t * math.sin(angle)

                if self._point_hits_static(px, py):
                    obs_dist = t
                    hit_static = 1.0
                    break

                if self._point_hits_dynamic(px, py):
                    obs_dist = t
                    hit_dynamic = 1.0
                    break

                if (not found_unclean) and self._point_unclean(px, py):
                    unclean_dist = t
                    hit_unclean = 1.0
                    found_unclean = True

                t += step

            hits[i, 0] = float(np.clip(obs_dist / max_dist, 0.0, 1.0))
            hits[i, 1] = hit_static
            hits[i, 2] = hit_dynamic
            hits[i, 3] = float(np.clip(unclean_dist / max_dist, 0.0, 1.0))
            hits[i, 4] = hit_unclean

        self.last_ray_hits = hits
        self.last_ray_summary = self._summarize_rays(hits)
        return hits

    def _summarize_rays(self, rays: np.ndarray) -> Dict[str, float]:
        center = len(rays) // 2
        span = max(1, len(rays) // 8)
        front = rays[max(0, center - span): min(len(rays), center + span + 1)]

        front_obs = float(np.min(front[:, 0])) if len(front) > 0 else 1.0
        min_obs = float(np.min(rays[:, 0])) if len(rays) > 0 else 1.0

        front_unclean = 1.0
        min_unclean = 1.0
        if np.any(front[:, 4] > 0.5):
            front_unclean = float(np.min(front[front[:, 4] > 0.5, 3]))
        if np.any(rays[:, 4] > 0.5):
            min_unclean = float(np.min(rays[rays[:, 4] > 0.5, 3]))

        return {
            "front_obstacle_dist": front_obs,
            "min_obstacle_dist": min_obs,
            "front_unclean_dist": front_unclean,
            "min_unclean_dist": min_unclean,
        }

    def _extract_local_patch(self) -> Tuple[np.ndarray, np.ndarray]:
        L = self.cfg.sensor.local_map_size
        half = L // 2
        gx = int(self.robot_x // self.grid)
        gy = int(self.robot_y // self.grid)

        clean_patch = np.zeros((L, L), dtype=np.float32)
        obstacle_patch = np.ones((L, L), dtype=np.float32)

        x0_src = max(0, gx - half)
        x1_src = min(self.grid_w, gx + half + 1)
        y0_src = max(0, gy - half)
        y1_src = min(self.grid_h, gy + half + 1)

        x0_dst = x0_src - (gx - half)
        y0_dst = y0_src - (gy - half)
        x1_dst = x0_dst + (x1_src - x0_src)
        y1_dst = y0_dst + (y1_src - y0_src)

        clean_patch[y0_dst:y1_dst,
                    x0_dst:x1_dst] = self.clean_map[y0_src:y1_src, x0_src:x1_src]
        obstacle_patch[y0_dst:y1_dst,
                       x0_dst:x1_dst] = self.obstacle_map[y0_src:y1_src, x0_src:x1_src]

        for obs in self.dynamic_obstacles:
            ogx = int(obs.x // self.grid)
            ogy = int(obs.y // self.grid)
            px = ogx - gx + half
            py = ogy - gy + half
            if 0 <= px < L and 0 <= py < L:
                obstacle_patch[py, px] = 1.0

        return clean_patch, obstacle_patch

    def _get_state_vector(self) -> np.ndarray:
        battery_ratio = self.battery / \
            max(self.cfg.robot.battery_capacity, 1e-6)
        dock_dx, dock_dy = self._dock_direction(self.robot_x, self.robot_y)
        dock_dx_norm = float(np.clip(dock_dx / max(self.W, 1), -1.0, 1.0))
        dock_dy_norm = float(np.clip(dock_dy / max(self.H, 1), -1.0, 1.0))
        dock_dist_norm = float(np.clip(
            math.hypot(dock_dx, dock_dy) /
            max(math.hypot(self.W, self.H), 1e-6), 0.0, 1.0
        ))

        summary = self.last_ray_summary if self.last_ray_summary else {
            "front_obstacle_dist": 1.0,
            "min_obstacle_dist": 1.0,
            "front_unclean_dist": 1.0,
            "min_unclean_dist": 1.0,
        }

        state = np.array([
            self.robot_x / self.W,
            self.robot_y / self.H,
            math.cos(self.robot_theta),
            math.sin(self.robot_theta),
            self.robot_v / max(self.cfg.robot.max_linear_speed, 1e-6),
            self.robot_w / max(self.cfg.robot.max_angular_speed, 1e-6),
            battery_ratio,
            1.0 if self._robot_on_dock() else 0.0,
            1.0 if self.charging_locked else 0.0,
            dock_dx_norm,
            dock_dy_norm,
            dock_dist_norm,
            summary["front_obstacle_dist"],
            summary["min_obstacle_dist"],
            summary["front_unclean_dist"],
            summary["min_unclean_dist"],
        ], dtype=np.float32)
        return state

    def _get_obs(self):
        rays = self._cast_rays()
        state = self._get_state_vector()
        local_clean_map, local_obstacle_map = self._extract_local_patch()

        if self.cfg.use_dict_observation:
            return {
                "state": state.astype(np.float32),
                "rays": rays.astype(np.float32),
                "local_clean_map": local_clean_map.astype(np.float32),
                "local_obstacle_map": local_obstacle_map.astype(np.float32),
            }

        return np.clip(np.concatenate([
            state.flatten(),
            rays.flatten(),
            local_clean_map.flatten(),
            local_obstacle_map.flatten(),
        ]).astype(np.float32), -1.0, 1.0)

    # =========================================================
    # Gym
    # =========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._np_random is None or seed is not None:
            self._seed(seed if seed is not None else self.cfg.seed)

        self.steps = 0
        self.clean_map.fill(0.0)
        self.dirt_map.fill(0.0)
        self.visit_map.fill(0.0)
        self.charging_locked = False
        self.last_done_reason = None
        self.prev_seen_unclean = False

        self._build_dock()
        self._build_valid_map()
        self._build_dirt_map()
        self._spawn_dynamic_obstacles()

        self._reset_robot_to_dock_center()
        self.battery = self.cfg.robot.battery_capacity

        self._mark_cleaned()
        self._mark_visited()
        self._cast_rays()
        self.prev_seen_unclean = bool(np.any(self.last_ray_hits[:, 4] > 0.5))

        obs = self._get_obs()
        info = {
            "coverage": self._coverage(),
            "battery": self.battery,
            "battery_ratio": 1.0,
            "on_dock": True,
            "charging_locked": False,
            "difficulty": self._difficulty(),
            "dynamic_obstacles": len(self.dynamic_obstacles),
            "dirty_cells": self.total_dirty_cells,
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

        prev_on_dock = self._robot_on_dock()
        prev_seen_unclean = self.prev_seen_unclean

        self._move_dynamic_obstacles()

        on_dock_before_move = self._robot_on_dock()

        if self.charging_locked:
            if on_dock_before_move:
                self.robot_v = 0.0
                self.robot_w = 0.0
                self.battery = min(self.cfg.robot.battery_capacity,
                                   self.battery + self.cfg.robot.charge_rate)
                reward_raw += self.cfg.reward.charging_wait_reward

                if self.battery >= self.cfg.robot.battery_capacity - 1e-8:
                    self.battery = self.cfg.robot.battery_capacity
                    self.charging_locked = False
                    self._reset_robot_to_dock_center()
            else:
                self.charging_locked = False
        else:
            a_v = float(action[0]) * self.cfg.robot.max_linear_acc
            a_w = float(action[1]) * self.cfg.robot.max_angular_acc

            self.robot_v += a_v * self.dt
            self.robot_w += a_w * self.dt

            self.robot_v *= self.cfg.robot.linear_drag
            self.robot_w *= self.cfg.robot.angular_drag

            self.robot_v = float(np.clip(
                self.robot_v,
                -self.cfg.robot.max_linear_speed,
                self.cfg.robot.max_linear_speed
            ))
            self.robot_w = float(np.clip(
                self.robot_w,
                -self.cfg.robot.max_angular_speed,
                self.cfg.robot.max_angular_speed
            ))

            new_theta = self._normalize_angle(
                self.robot_theta + self.robot_w * self.dt)
            new_x = self.robot_x + self.robot_v * math.cos(new_theta) * self.dt
            new_y = self.robot_y + self.robot_v * math.sin(new_theta) * self.dt

            coll_static = self._robot_collides_static(new_x, new_y)
            coll_dynamic = self._robot_collides_dynamic(new_x, new_y)

            if coll_static or coll_dynamic:
                reward_raw += self.cfg.reward.collision_penalty
                terminated = True
                done_reason = "collision_static" if coll_static else "collision_dynamic"
            else:
                self.robot_x = new_x
                self.robot_y = new_y
                self.robot_theta = new_theta

            move_ratio = abs(self.robot_v) / \
                max(self.cfg.robot.max_linear_speed, 1e-6)
            turn_ratio = abs(self.robot_w) / \
                max(self.cfg.robot.max_angular_speed, 1e-6)
            battery_cost = (
                self.cfg.robot.battery_per_step +
                self.cfg.robot.battery_move_scale * move_ratio +
                self.cfg.robot.battery_turn_scale * turn_ratio
            )
            self.battery = max(0.0, self.battery - battery_cost)

        on_dock = self._robot_on_dock()

        newly_cleaned, revisited = self._mark_cleaned()
        first_visit = self._mark_visited()

        self._cast_rays()
        summary = self.last_ray_summary
        cur_seen_unclean = bool(np.any(self.last_ray_hits[:, 4] > 0.5))
        self.prev_seen_unclean = cur_seen_unclean

        # 主奖励
        reward_raw += self.cfg.reward.step_penalty
        reward_raw += newly_cleaned * self.cfg.reward.new_clean_cell_reward
        reward_raw += first_visit * self.cfg.reward.first_visit_reward
        reward_raw += revisited * self.cfg.reward.revisit_penalty

        # 首次发现未清扫污渍
        if (not prev_seen_unclean) and cur_seen_unclean:
            reward_raw += self.cfg.reward.discover_dirt_reward

        # 回桩奖励
        if (not prev_on_dock) and on_dock:
            remaining = self.battery / \
                max(self.cfg.robot.battery_capacity, 1e-6)

            if remaining <= self.cfg.robot.low_battery_threshold:
                low_factor = 1.0 - remaining / \
                    max(self.cfg.robot.low_battery_threshold, 1e-6)
                reward_raw += (
                    self.cfg.reward.dock_return_base_reward +
                    self.cfg.reward.dock_return_low_battery_bonus * low_factor
                )
            else:
                reward_raw += self.cfg.reward.early_dock_penalty * remaining

            self.charging_locked = True
            self.robot_v = 0.0
            self.robot_w = 0.0

        if self.battery <= 0.0 and not terminated:
            self.robot_v = 0.0
            self.robot_w = 0.0
            reward_raw += self.cfg.reward.out_of_power_penalty
            terminated = True
            done_reason = "out_of_power"

        if self.steps >= self.max_steps and not terminated:
            reward_raw += self.cfg.reward.timeout_penalty
            truncated = True
            done_reason = "timeout"

        self.last_done_reason = done_reason

        reward = float(reward_raw * self.cfg.reward.reward_scale)
        obs = self._get_obs()
        info = {
            "coverage": self._coverage(),
            "battery": self.battery,
            "battery_ratio": self.battery / max(self.cfg.robot.battery_capacity, 1e-6),
            "steps": self.steps,
            "newly_cleaned": newly_cleaned,
            "revisited": revisited,
            "first_visit": first_visit,
            "seen_unclean": cur_seen_unclean,
            "on_dock": on_dock,
            "charging_locked": self.charging_locked,
            "front_obstacle_dist": summary["front_obstacle_dist"],
            "min_obstacle_dist": summary["min_obstacle_dist"],
            "front_unclean_dist": summary["front_unclean_dist"],
            "min_unclean_dist": summary["min_unclean_dist"],
            "difficulty": self._difficulty(),
            "dynamic_obstacles": len(self.dynamic_obstacles),
            "dirty_cells": self.total_dirty_cells,
            "reward_raw": float(reward_raw),
            "reward": reward,
            "done_reason": done_reason,
        }

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    # =========================================================
    # render
    # =========================================================
    def _init_pygame(self) -> None:
        if not self._render_enabled:
            return
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((self.W, self.H))
                pygame.display.set_caption(self.cfg.render.window_caption)
            self.clock = pygame.time.Clock()
            self.canvas = pygame.Surface((self.W, self.H))

    def render(self):
        if not self._render_enabled or self.render_mode is None:
            return None

        self._init_pygame()
        self.canvas.fill((238, 238, 238))

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                if self.obstacle_map[gy, gx] > 0.5:
                    continue

                x = gx * self.grid
                y = gy * self.grid
                rect = pygame.Rect(x, y, self.grid, self.grid)

                if self.dirt_map[gy, gx] > 0.5:
                    color = (205, 240, 205) if self.clean_map[gy, gx] > 0.5 else (
                        220, 190, 145)
                else:
                    color = (245, 245, 245)

                pygame.draw.rect(self.canvas, color, rect)

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

        for obs in self.obstacles:
            color = (90, 90, 90) if obs.kind == "block" else (70, 70, 110)
            pygame.draw.rect(self.canvas, color, obs.rect)

        pygame.draw.rect(self.canvas, (80, 150, 255), self.dock_rect)

        for obs in self.dynamic_obstacles:
            pygame.draw.circle(self.canvas, (255, 165, 60),
                               (int(obs.x), int(obs.y)), int(obs.radius))

        robot_pos = (int(self.robot_x), int(self.robot_y))
        pygame.draw.circle(self.canvas, (220, 80, 80),
                           robot_pos, int(self.cfg.robot.radius))
        hx = self.robot_x + self.cfg.robot.radius * math.cos(self.robot_theta)
        hy = self.robot_y + self.cfg.robot.radius * math.sin(self.robot_theta)
        pygame.draw.line(self.canvas, (255, 255, 255),
                         robot_pos, (int(hx), int(hy)), 3)

        if self.cfg.render.draw_rays and self.last_ray_hits is not None:
            num_rays = self.cfg.sensor.num_rays
            max_dist = self.cfg.sensor.ray_max_distance
            fov = math.radians(self.cfg.sensor.ray_fov_deg)
            angles = np.array([self.robot_theta], dtype=np.float32) if num_rays == 1 else np.linspace(
                self.robot_theta - fov / 2.0,
                self.robot_theta + fov / 2.0,
                num_rays,
            )

            for i, angle in enumerate(angles):
                obs_dist = self.last_ray_hits[i, 0] * max_dist
                hit_static = self.last_ray_hits[i, 1] > 0.5
                hit_dynamic = self.last_ray_hits[i, 2] > 0.5
                unclean_dist = self.last_ray_hits[i, 3] * max_dist
                hit_unclean = self.last_ray_hits[i, 4] > 0.5

                end_obs = (
                    int(self.robot_x + obs_dist * math.cos(angle)),
                    int(self.robot_y + obs_dist * math.sin(angle))
                )

                if hit_dynamic:
                    color_obs = (255, 180, 60)
                elif hit_static:
                    color_obs = (120, 170, 255)
                else:
                    color_obs = (200, 200, 200)

                pygame.draw.line(self.canvas, color_obs, robot_pos, end_obs, 1)

                if hit_unclean:
                    px = int(self.robot_x + unclean_dist * math.cos(angle))
                    py = int(self.robot_y + unclean_dist * math.sin(angle))
                    pygame.draw.circle(
                        self.canvas, (255, 180, 60), (px, py), 2)

        if self.cfg.render.draw_status_text:
            coverage = self._coverage()
            battery_ratio = self.battery / \
                max(self.cfg.robot.battery_capacity, 1e-6)
            text = (
                f"step={self.steps}  cov={coverage:.3f}  bat={self.battery:.1f}({battery_ratio:.2f})  "
                f"dock={int(self._robot_on_dock())}  lock={int(self.charging_locked)}  "
                f"dirty={self.total_dirty_cells}  diff={self._difficulty()}"
            )
            font = pygame.font.SysFont("consolas", 18)
            self.canvas.blit(font.render(text, True, (20, 20, 20)), (10, 10))

            if self.last_done_reason is not None:
                self.canvas.blit(font.render(
                    f"done={self.last_done_reason}", True, (20, 20, 20)), (10, 34))

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


def run_random_demo(steps: int = 400, seed: int = 42, difficulty: str = "medium"):
    cfg = EnvConfig(difficulty=difficulty)
    env = CleaningRobotEnv(cfg=cfg, render_mode="human")
    obs, info = env.reset(seed=seed)
    print("reset:", info)

    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample())
        if info["steps"] % 50 == 0:
            print(
                f"step={info['steps']}, reward={reward:.3f}, cov={info['coverage']:.3f}, "
                f"bat={info['battery']:.1f}, dirty={info['dirty_cells']}, "
                f"new={info['newly_cleaned']}, revisit={info['revisited']}, "
                f"first={info['first_visit']}, seen_unclean={int(info['seen_unclean'])}, "
                f"diff={info['difficulty']}, done={info['done_reason']}"
            )
        if terminated or truncated:
            print("episode end:", info)
            break
    env.close()


if __name__ == "__main__":
    run_random_demo(difficulty="easy")
