from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RobotConfig:
    """
    机器人参数
    """
    radius: float = 12.0
    max_linear_speed: float = 90.0          # 像素/秒
    max_angular_speed: float = 2.8          # 弧度/秒

    battery_capacity: float = 1000.0
    battery_per_step: float = 1.0           # 每步基础耗电
    battery_move_scale: float = 0.6         # 速度越快，额外耗电越多

    cleaning_radius: float = 20.0           # 清扫半径
    dock_recharge_rate: float = 10.0        # 在充电桩内每步回充电量

    low_battery_threshold: float = 0.25     # 低电阈值（比例）
    critical_battery_threshold: float = 0.12

    spawn_safe_radius: float = 100.0        # 初始出生点与充电桩最小安全距离


@dataclass
class MapConfig:
    """
    地图参数
    """
    width: int = 900
    height: int = 650
    border_thickness: int = 8

    clean_grid_size: int = 20

    obstacle_count_range: Tuple[int, int] = (4, 8)
    obstacle_min_size: Tuple[int, int] = (60, 50)
    obstacle_max_size: Tuple[int, int] = (140, 120)
    obstacle_margin: int = 24

    dock_size: int = 34
    dock_pos: Tuple[int, int] = (60, 60)
    dock_safe_clearance: int = 72           # dock 周围预留安全空间

    min_cleanable_ratio: float = 0.50       # 至少保证一定比例可清扫区域
    max_obstacle_ratio: float = 0.42        # 障碍比例上限，避免地图过于拥挤
    min_reachable_ratio_from_dock: float = 0.75
    require_dock_reachable_component: bool = True
    require_spawn_reachable_to_dock: bool = True


@dataclass
class SensorConfig:
    """
    传感器参数
    """
    num_rays: int = 21
    ray_max_distance: float = 180.0
    ray_fov_deg: float = 220.0
    ray_step: float = 4.0

    # 局部窗口（以机器人为中心，从全局清扫网格裁切）
    local_map_size: int = 11

    # 是否显式加入 dock 局部通道
    use_local_dock_map: bool = True

    # 射线摘要配置
    front_ray_half_span: int = 2            # 以中心射线为基准，左右各取几条作为“前方”
    dynamic_hit_distance_ratio: float = 0.35  # 前方此距离内若有动态障碍，front_has_dynamic=1


@dataclass
class DynamicObstacleConfig:
    """
    动态障碍参数
    """
    enabled: bool = True
    count_range: Tuple[int, int] = (2, 4)
    radius_range: Tuple[int, int] = (10, 16)
    speed_range: Tuple[float, float] = (25.0, 55.0)

    # 与静态障碍 / 边界 / 机器人发生碰撞时反弹
    max_try_spawn: int = 300

    # 出生时与 dock / 机器人 / 静态障碍额外保持距离
    dock_avoid_radius: float = 70.0
    robot_avoid_radius: float = 100.0
    pair_safe_gap: float = 8.0


@dataclass
class RewardConfig:
    """
    奖励函数参数

    设计目标：
    - 高电量：主打清扫，不鼓励占桩
    - 低电量：允许回桩，并奖励有效回充
    - 抑制刷分策略：原地不动 / 长时间无进展 / 高电量停桩
    """
    # 基础
    step_penalty: float = -0.02

    # 清扫主奖励
    new_clean_cell_reward: float = 1.0
    coverage_gain_reward: float = 12.0

    # 避障 / 失败
    collision_static_penalty: float = -15.0
    collision_dynamic_penalty: float = -18.0
    battery_empty_penalty: float = -20.0

    # 回桩 / 充电
    low_battery_to_dock_progress_reward: float = 8.0
    low_battery_away_from_dock_penalty: float = -0.08
    dock_contact_reward: float = 1.5
    recharge_reward_scale: float = 0.10
    low_battery_return_success_bonus: float = 10.0

    # 高电量占桩抑制
    non_low_battery_dock_penalty: float = -0.30
    high_battery_dock_penalty: float = -0.80
    high_battery_threshold_for_dock_penalty: float = 0.60

    # 停滞 / 无进展抑制
    low_battery_clean_reward_scale: float = 0.20

    idle_linear_speed_ratio_threshold: float = 0.08
    idle_angular_speed_ratio_threshold: float = 0.08
    idle_step_threshold: int = 18
    idle_penalty: float = -0.25

    no_progress_step_threshold: int = 28
    no_progress_penalty: float = -0.40

    dock_idle_step_threshold: int = 12
    dock_idle_penalty: float = -0.60

    # 为了防止一开局去充电桩磨时间，可在覆盖率很低时额外压制占桩
    early_stage_coverage_threshold: float = 0.08
    early_stage_dock_penalty: float = -0.50

    # 成功 / 截断
    full_clean_bonus: float = 25.0
    timeout_penalty: float = -3.0

    # 整体 reward 缩放，减轻 critic 压力
    reward_scale: float = 0.1


@dataclass
class RenderConfig:
    """
    渲染参数
    """
    render_fps: int = 30
    window_caption: str = "Cleaning Robot Multi-Task Env"

    draw_grid: bool = True
    draw_status_text: bool = True
    draw_rays: bool = True
    draw_local_patch_outline: bool = False


@dataclass
class CurriculumConfig:
    """
    训练阶段配置
    可通过 EnvConfig.training_mode 切换
    """
    easy_dynamic_enabled: bool = False
    mid_dynamic_enabled: bool = False
    full_dynamic_enabled: bool = True

    easy_battery_capacity_scale: float = 1.35
    mid_battery_capacity_scale: float = 1.10
    full_battery_capacity_scale: float = 1.00

    easy_obstacle_scale: float = 0.70
    mid_obstacle_scale: float = 0.90
    full_obstacle_scale: float = 1.00


@dataclass
class EnvConfig:
    """
    总配置
    """
    robot: RobotConfig = field(default_factory=RobotConfig)
    map_cfg: MapConfig = field(default_factory=MapConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    dynamic_obstacle: DynamicObstacleConfig = field(
        default_factory=DynamicObstacleConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    dt: float = 0.1
    max_steps: int = 2200
    seed: int | None = 42

    # 训练阶段：
    # - baseline_easy
    # - baseline_mid
    # - baseline_full
    training_mode: str = "baseline_easy"

    # True: 返回 dict observation
    # False: 返回扁平 observation
    use_dict_observation: bool = True
