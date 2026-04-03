from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MapConfig:
    width: int = 1280
    height: int = 1280
    border_thickness: int = 8

    clean_grid_size: int = 16
    dock_size: int = 56
    dock_pos: Tuple[int, int] = (40, 40)

    # 静态障碍数量
    block_obstacle_count_range: Tuple[int, int] = (2, 4)
    wall_obstacle_count_range: Tuple[int, int] = (1, 3)

    # 方块障碍尺寸
    block_min_size: Tuple[int, int] = (40, 40)
    block_max_size: Tuple[int, int] = (90, 90)

    # 墙体障碍尺寸
    wall_thickness_range: Tuple[int, int] = (16, 28)
    wall_length_range: Tuple[int, int] = (120, 220)

    obstacle_margin: int = 18
    dock_safe_clearance: int = 72
    min_cleanable_ratio: float = 0.60


@dataclass
class DirtConfig:
    enabled: bool = True

    # 污渍聚落数量
    cluster_count_range: Tuple[int, int] = (4, 8)

    # 聚落半径（单位：grid cell）
    cluster_radius_range: Tuple[int, int] = (4, 9)

    # 聚落密度，整体提高
    cluster_density_range: Tuple[float, float] = (0.72, 0.95)

    # 距离 dock 的安全距离（像素）
    spawn_safe_distance_to_dock: float = 120.0

    # 总污渍占可通行区域比例
    min_dirty_ratio: float = 0.12
    max_dirty_ratio: float = 0.40

    # 生成重试次数
    max_retry: int = 60


@dataclass
class DynamicObstacleConfig:
    enabled: bool = True
    count_range: Tuple[int, int] = (1, 2)
    radius_range: Tuple[int, int] = (10, 16)
    speed_range: Tuple[float, float] = (35.0, 65.0)
    spawn_safe_distance_to_dock: float = 100.0
    pair_safe_gap: float = 18.0
    max_try_spawn: int = 120


@dataclass
class RobotConfig:
    radius: float = 14.0
    cleaning_radius: float = 24.0

    # 加速度模型
    max_linear_speed: float = 80.0
    max_angular_speed: float = 2.4
    max_linear_acc: float = 120.0
    max_angular_acc: float = 4.0

    linear_drag: float = 0.92
    angular_drag: float = 0.88

    # 电池
    battery_capacity: float = 100.0
    battery_per_step: float = 0.08
    battery_move_scale: float = 0.10
    battery_turn_scale: float = 0.04

    charge_rate: float = 0.55
    low_battery_threshold: float = 0.22


@dataclass
class SensorConfig:
    num_rays: int = 31
    ray_fov_deg: float = 220.0
    ray_max_distance: float = 150.0
    ray_step: float = 4.0

    local_map_size: int = 15


@dataclass
class RewardConfig:
    step_penalty: float = -0.005

    # 主奖励：只针对污渍清扫
    new_clean_cell_reward: float = 0.45

    # 探索奖励适当减弱
    first_visit_reward: float = 0.02

    # 重复覆盖轻微惩罚
    revisit_penalty: float = -0.012

    # 新增：首次看到未清扫污渍
    discover_dirt_reward: float = 0.05

    # 回桩
    dock_return_base_reward: float = 0.10
    dock_return_low_battery_bonus: float = 0.35
    early_dock_penalty: float = -0.08

    # 电量耗尽
    out_of_power_penalty: float = -5.0

    # 充电等待成本
    charging_wait_reward: float = -0.001

    # 失败惩罚
    collision_penalty: float = -4.0
    timeout_penalty: float = 0.0

    reward_scale: float = 1.0


@dataclass
class RenderConfig:
    window_caption: str = "Simplified Cleaning Robot Env"
    render_fps: int = 30
    draw_grid: bool = True
    draw_rays: bool = True
    draw_status_text: bool = True


@dataclass
class EnvConfig:
    seed: int = 42
    dt: float = 0.10
    max_steps: int = 2200

    use_dict_observation: bool = True

    # easy / medium / hard
    difficulty: str = "medium"

    map_cfg: MapConfig = field(default_factory=MapConfig)
    dirt: DirtConfig = field(default_factory=DirtConfig)
    dynamic_obstacle: DynamicObstacleConfig = field(
        default_factory=DynamicObstacleConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
