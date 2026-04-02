import os
import json
import time
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from env_cfg import EnvConfig
from cleaning_robot_env import CleaningRobotEnv


"""
=========================================================
moeppobaseline.py
=========================================================

版本日志
---------------------------------------------------------
v1:
- 基础 MoE-PPO
- 4 expert: global_clean / local_avoid / return_dock / general
- simple MLP gating
- 问题：general 明显塌缩吞噬其余 expert

v2:
- 改为 3 expert: global_clean / local_avoid / return_dock
- 去掉 general
- 增加 anti-collapse:
  1) gate_balance_loss
  2) gate_entropy regularization
- 问题：塌缩缓解，但 return_dock 往往学成“保守通用 expert”，
  而不是真正的低电量回桩 expert

v3（当前版）:
- 保留 3 expert
- 保留 anti-collapse
- 新增“场景弱引导 gate loss”
  * 低电量 -> 轻微鼓励 return_dock
  * 近障碍 -> 轻微鼓励 local_avoid
  * 普通状态 -> 轻微鼓励 global_clean
- 新增分场景 gate 日志:
  * 全 batch gate
  * low-battery 子集 gate
  * near-obstacle 子集 gate

设计原则
---------------------------------------------------------
1. 仍然是 PPO 主体，不改 PPO 训练框架
2. gate 引导是“弱监督”，不是硬规则切换
3. expert 分工来自:
   - PPO 回报
   - anti-collapse 正则
   - 轻微场景语义对齐
4. 尽量避免依赖 state 向量内部不透明字段含义，
   引导主要依赖:
   - info["battery_ratio"]
   - obs["rays"]
   - obs["local_obstacle_map"]
   - obs["local_dock_map"]（若存在）
=========================================================
"""


# =========================================================
# 基础工具
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))


def obs_to_tensor(obs_dict: dict, device: torch.device) -> dict:
    return {
        k: torch.as_tensor(v, dtype=torch.float32, device=device)
        for k, v in obs_dict.items()
    }


def stack_local_maps(obs_dict: dict) -> torch.Tensor:
    """
    输入:
        local_clean_map: [B, H, W]
        local_obstacle_map: [B, H, W]
        optional local_dock_map: [B, H, W]
    输出:
        [B, C, H, W]
    """
    maps = [obs_dict["local_clean_map"], obs_dict["local_obstacle_map"]]
    if "local_dock_map" in obs_dict:
        maps.append(obs_dict["local_dock_map"])
    return torch.stack(maps, dim=1)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_mean(x, default=0.0) -> float:
    if len(x) == 0:
        return float(default)
    return float(np.mean(x))


def extract_final_info_list(infos: dict, num_envs: int):
    """
    兼容不同 Gymnasium 版本的 vector infos:
    - infos["final_info"] + infos["_final_info"]
    - 仅有 infos["final_info"]
    """
    out = [None] * num_envs

    if "final_info" not in infos:
        return out

    final_infos = infos["final_info"]

    if "_final_info" in infos:
        mask = infos["_final_info"]
        for i in range(num_envs):
            if bool(mask[i]):
                out[i] = final_infos[i]
        return out

    for i in range(num_envs):
        fi = final_infos[i]
        if fi is not None:
            out[i] = fi
    return out


def get_info_array(info: dict, key: str, num_envs: int, default: float = 0.0):
    """
    从 vector info 里安全取长度为 num_envs 的数组
    """
    if info is None or key not in info:
        return np.full(num_envs, default, dtype=np.float32)

    value = info[key]
    arr = np.asarray(value)

    if arr.ndim == 0:
        return np.full(num_envs, float(arr), dtype=np.float32)

    if len(arr) != num_envs:
        return np.full(num_envs, default, dtype=np.float32)

    return arr.astype(np.float32)


# =========================================================
# Tanh Gaussian
# =========================================================
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
EPS = 1e-6


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + EPS, 1.0 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class SquashedNormal:
    """
    tanh-squashed Gaussian
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.std = torch.exp(self.log_std)
        self.normal = torch.distributions.Normal(self.mean, self.std)

    def sample(self):
        u = self.normal.rsample()
        a = torch.tanh(u)
        log_prob = self.normal.log_prob(u) - torch.log(1.0 - a.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1)
        return a, log_prob

    def log_prob(self, action: torch.Tensor):
        u = atanh(action)
        log_prob = self.normal.log_prob(
            u) - torch.log(1.0 - action.pow(2) + EPS)
        return log_prob.sum(dim=-1)

    def entropy_proxy(self):
        """
        tanh 后精确 entropy 不直接写，这里用 unsquashed normal entropy 近似监控
        """
        return self.normal.entropy().sum(dim=-1)

    def mode(self):
        return torch.tanh(self.mean)


# =========================================================
# 编码器
# =========================================================
class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RayEncoder(nn.Module):
    def __init__(self, num_rays: int, ray_dim: int = 3, out_dim: int = 128):
        super().__init__()
        in_dim = num_rays * ray_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, out_dim),
            nn.Tanh(),
        )

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        x = rays.reshape(rays.shape[0], -1)
        return self.net(x)


class LocalMapEncoder(nn.Module):
    def __init__(self, in_channels: int, local_map_size: int, out_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, local_map_size, local_map_size)
            flat_dim = self.cnn(dummy).shape[1]

        self.proj = nn.Sequential(
            nn.Linear(flat_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.cnn(x))


# =========================================================
# MoE 网络
# =========================================================
class ExpertHead(nn.Module):
    """
    每个 expert 对共享特征再做一层小 MLP，然后输出:
    - actor mean
    - actor log_std
    - critic value
    """

    def __init__(self, in_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        mean = self.actor_mean(h)
        log_std = self.actor_log_std(h)
        value = self.critic(h).squeeze(-1)
        return mean, log_std, value


class MoeActorCritic(nn.Module):
    """
    结构:
        obs -> shared encoders -> shared backbone -> gating + experts

    这里使用 soft gating:
        gate_weights = softmax(gating(feat))

    最终策略参数与 value 都由多 expert 加权融合得到
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_dim: int,
        expert_names=None,
        shared_dim: int = 256,
        expert_hidden_dim: int = 128,
    ):
        super().__init__()
        assert isinstance(
            obs_space, gym.spaces.Dict), "当前版本要求 dict observation"

        if expert_names is None:
            expert_names = ["global_clean", "local_avoid", "return_dock"]

        self.expert_names = expert_names
        self.num_experts = len(expert_names)
        self.action_dim = action_dim

        state_dim = obs_space["state"].shape[0]
        num_rays = obs_space["rays"].shape[0]
        ray_dim = obs_space["rays"].shape[1]
        local_map_size = obs_space["local_clean_map"].shape[0]
        in_channels = 2 + (1 if "local_dock_map" in obs_space.spaces else 0)

        self.state_encoder = StateEncoder(state_dim, 128)
        self.ray_encoder = RayEncoder(num_rays, ray_dim, 128)
        self.map_encoder = LocalMapEncoder(in_channels, local_map_size, 128)

        self.backbone = nn.Sequential(
            nn.Linear(384, shared_dim),
            nn.Tanh(),
            nn.Linear(shared_dim, shared_dim),
            nn.Tanh(),
        )

        # gating 网络：简单 MLP
        self.gating = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.Tanh(),
            nn.Linear(128, self.num_experts),
        )

        self.experts = nn.ModuleList([
            ExpertHead(shared_dim, expert_hidden_dim, action_dim)
            for _ in range(self.num_experts)
        ])

    def encode(self, obs_dict: dict) -> torch.Tensor:
        state_feat = self.state_encoder(obs_dict["state"])
        ray_feat = self.ray_encoder(obs_dict["rays"])
        map_feat = self.map_encoder(stack_local_maps(obs_dict))
        x = torch.cat([state_feat, ray_feat, map_feat], dim=-1)
        return self.backbone(x)

    def get_gate_weights(self, feat: torch.Tensor):
        gate_logits = self.gating(feat)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        return gate_logits, gate_weights

    def mixture_forward(self, obs_dict: dict):
        feat = self.encode(obs_dict)                      # [B, D]
        gate_logits, gate_weights = self.get_gate_weights(feat)  # [B, E]

        expert_means = []
        expert_log_stds = []
        expert_values = []

        for expert in self.experts:
            mean_i, log_std_i, value_i = expert(feat)
            expert_means.append(mean_i)
            expert_log_stds.append(log_std_i)
            expert_values.append(value_i)

        # [B, E, A]
        expert_means = torch.stack(expert_means, dim=1)
        expert_log_stds = torch.stack(expert_log_stds, dim=1)
        # [B, E]
        expert_values = torch.stack(expert_values, dim=1)

        gate_w_act = gate_weights.unsqueeze(-1)  # [B, E, 1]

        mean = (expert_means * gate_w_act).sum(dim=1)
        log_std = (expert_log_stds * gate_w_act).sum(dim=1)
        value = (expert_values * gate_weights).sum(dim=1)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        aux = {
            "feat": feat,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "expert_means": expert_means,
            "expert_log_stds": expert_log_stds,
            "expert_values": expert_values,
        }
        return mean, log_std, value, aux

    def get_dist_and_value(self, obs_dict: dict):
        mean, log_std, value, aux = self.mixture_forward(obs_dict)
        dist = SquashedNormal(mean, log_std)
        return dist, value, aux

    def get_action_value(self, obs_dict: dict, action=None):
        dist, value, aux = self.get_dist_and_value(obs_dict)
        if action is None:
            action, log_prob = dist.sample()
        else:
            action = torch.clamp(action, -1.0 + EPS, 1.0 - EPS)
            log_prob = dist.log_prob(action)
        entropy = dist.entropy_proxy()
        return action, log_prob, entropy, value, aux

    def act_deterministic(self, obs_dict: dict):
        dist, _, aux = self.get_dist_and_value(obs_dict)
        return dist.mode(), aux


# =========================================================
# 环境构建
# =========================================================
def make_env(cfg: EnvConfig, seed: int, idx: int):
    def thunk():
        env = CleaningRobotEnv(cfg=cfg, render_mode=None)
        env.reset(seed=seed + idx)
        return env
    return thunk


# =========================================================
# curriculum 阶段
# =========================================================
def build_stage_cfg(base_cfg: EnvConfig, training_mode: str) -> EnvConfig:
    cfg = EnvConfig()
    cfg.robot = base_cfg.robot
    cfg.map_cfg = base_cfg.map_cfg
    cfg.sensor = base_cfg.sensor
    cfg.dynamic_obstacle = base_cfg.dynamic_obstacle
    cfg.reward = base_cfg.reward
    cfg.render = base_cfg.render
    cfg.curriculum = base_cfg.curriculum

    cfg.dt = base_cfg.dt
    cfg.max_steps = base_cfg.max_steps
    cfg.seed = base_cfg.seed
    cfg.use_dict_observation = base_cfg.use_dict_observation
    cfg.training_mode = training_mode
    return cfg


def stage_name_by_update(update: int, num_updates: int) -> str:
    ratio = update / max(num_updates, 1)
    if ratio <= 0.35:
        return "baseline_easy"
    elif ratio <= 0.70:
        return "baseline_mid"
    return "baseline_full"


# =========================================================
# gate 正则与弱引导
# =========================================================
def compute_gate_regularization(
    gate_weights: torch.Tensor,
    num_experts: int,
):
    """
    gate_weights: [B, E]

    返回:
        balance_loss: batch 平均路由接近均匀分布
        gate_entropy: 每个样本的 gating 熵（越大表示越不容易早期塌缩）
        usage:        batch 内平均 usage
    """
    usage = gate_weights.mean(dim=0)  # [E]
    target = torch.full_like(usage, 1.0 / num_experts)

    balance_loss = ((usage - target) ** 2).mean()
    gate_entropy = - \
        (gate_weights * torch.log(gate_weights + EPS)).sum(dim=-1).mean()

    return balance_loss, gate_entropy, usage


def derive_guidance_targets(
    obs: dict,
    info: dict,
    cfg: EnvConfig,
    expert_names,
    avoid_ray_threshold: float = 0.18,
    avoid_map_density_threshold: float = 0.20,
    dock_visible_battery_margin: float = 0.08,
):
    """
    基于“当前状态”生成场景弱引导目标。
    这是 v3 的关键改动。

    规则优先级:
        1) 低电量 -> return_dock
        2) 近障碍 -> local_avoid
        3) 其余 -> global_clean

    注意:
    - 这是弱引导，不是硬切换
    - 输出的是 soft target distribution，而不是 one-hot
    - 训练时只作为一个小权重 gate loss

    返回:
        guide_targets: [N, 3]
        low_battery_mask: [N]
        near_obstacle_mask: [N]
        dock_visible_mask: [N]
    """
    num_envs = obs["state"].shape[0]

    # 约定 expert 顺序
    idx_global = expert_names.index("global_clean")
    idx_avoid = expert_names.index("local_avoid")
    idx_dock = expert_names.index("return_dock")

    # -------------------------
    # battery
    # -------------------------
    battery_ratio = get_info_array(
        info, "battery_ratio", num_envs, default=1.0)
    low_battery_mask = battery_ratio <= float(cfg.robot.low_battery_threshold)

    # -------------------------
    # dock visible
    # -------------------------
    if "local_dock_map" in obs:
        dock_visible_mask = np.asarray(
            obs["local_dock_map"].reshape(num_envs, -1).sum(axis=1) > 1e-6,
            dtype=bool,
        )
    else:
        dock_visible_mask = np.zeros(num_envs, dtype=bool)

    # 对“稍低电量 + 看见 dock”也可以适当鼓励 dock expert
    near_low_battery_mask = battery_ratio <= float(
        cfg.robot.low_battery_threshold + dock_visible_battery_margin)
    dock_guided_mask = np.logical_or(low_battery_mask, np.logical_and(
        near_low_battery_mask, dock_visible_mask))

    # -------------------------
    # near obstacle
    # -------------------------
    # 假设 rays[..., 0] 是最主要的距离通道（通常是归一化距离）
    # 这是相对保守、常见的约定；不依赖 state 内部字段顺序
    rays = obs["rays"]
    ray_min = np.min(rays[..., 0], axis=1)

    obstacle_density = np.mean(
        obs["local_obstacle_map"].reshape(num_envs, -1), axis=1)
    near_obstacle_mask = np.logical_or(
        ray_min < avoid_ray_threshold,
        obstacle_density > avoid_map_density_threshold,
    )

    # -------------------------
    # soft targets
    # -------------------------
    # 三类目标分布都不是 one-hot，避免 gate 被硬掰断
    target_global = np.zeros((num_envs, len(expert_names)), dtype=np.float32)
    target_avoid = np.zeros((num_envs, len(expert_names)), dtype=np.float32)
    target_dock = np.zeros((num_envs, len(expert_names)), dtype=np.float32)

    # 普通清扫：主推 global_clean，但允许其余 expert 少量参与
    target_global[:, idx_global] = 0.70
    target_global[:, idx_avoid] = 0.15
    target_global[:, idx_dock] = 0.15

    # 近障碍：主推 local_avoid
    target_avoid[:, idx_global] = 0.15
    target_avoid[:, idx_avoid] = 0.70
    target_avoid[:, idx_dock] = 0.15

    # 低电量 / 可见 dock 且电量偏低：主推 return_dock
    target_dock[:, idx_global] = 0.10
    target_dock[:, idx_avoid] = 0.10
    target_dock[:, idx_dock] = 0.80

    guide_targets = target_global.copy()

    # 规则优先级：dock > avoid > global
    guide_targets[near_obstacle_mask] = target_avoid[near_obstacle_mask]
    guide_targets[dock_guided_mask] = target_dock[dock_guided_mask]

    return (
        guide_targets.astype(np.float32),
        low_battery_mask.astype(np.float32),
        near_obstacle_mask.astype(np.float32),
        dock_visible_mask.astype(np.float32),
    )


def compute_gate_guidance_loss(gate_weights: torch.Tensor, guide_targets: torch.Tensor):
    """
    gate_weights:  [B, E]
    guide_targets: [B, E]

    使用 soft target cross-entropy:
        L = - sum_i target_i * log(gate_i)

    这是“弱场景语义对齐”损失。
    """
    return -(guide_targets * torch.log(gate_weights + EPS)).sum(dim=-1).mean()


# =========================================================
# 评估
# =========================================================
@torch.no_grad()
def evaluate_policy(
    model: MoeActorCritic,
    device: torch.device,
    cfg: EnvConfig,
    eval_episodes: int = 8,
    render: bool = False,
) -> dict:
    returns = []
    coverages = []

    collision_count = 0
    battery_empty_count = 0
    full_clean_count = 0
    timeout_count = 0
    return_success_count = 0

    mean_idle_steps = []
    mean_no_progress_steps = []
    mean_dock_idle_steps = []

    gate_weight_records = []
    low_battery_gate_records = []
    near_obstacle_gate_records = []

    for ep in range(eval_episodes):
        env = CleaningRobotEnv(
            cfg=cfg, render_mode="human" if render else None)
        obs, info = env.reset(seed=(cfg.seed or 42) + 100000 + ep)

        done = False
        ep_ret = 0.0
        low_battery_entered = False
        returned_on_dock = False

        ep_idle_sum = 0.0
        ep_no_progress_sum = 0.0
        ep_dock_idle_sum = 0.0
        ep_len = 0

        ep_gate_weights = []
        ep_low_gate = []
        ep_near_gate = []

        current_info = info

        while not done:
            # 当前状态下做场景分析
            _, low_mask, near_mask, _ = derive_guidance_targets(
                obs={k: np.expand_dims(v, 0) for k, v in obs.items()},
                info={k: np.asarray([v]) if np.isscalar(v) else np.asarray(
                    [v]) for k, v in current_info.items()} if current_info is not None else {},
                cfg=cfg,
                expert_names=model.expert_names,
            )

            obs_t = {
                k: torch.as_tensor(np.expand_dims(
                    v, 0), dtype=torch.float32, device=device)
                for k, v in obs.items()
            }

            action_t, aux = model.act_deterministic(obs_t)
            action = action_t.cpu().numpy()[0]

            gate_now = aux["gate_weights"].cpu().numpy()[0]
            ep_gate_weights.append(gate_now)

            if bool(low_mask[0]):
                ep_low_gate.append(gate_now)
            if bool(near_mask[0]):
                ep_near_gate.append(gate_now)

            obs, reward, terminated, truncated, info = env.step(action)
            current_info = info

            ep_ret += reward
            ep_len += 1

            ep_idle_sum += float(info.get("idle_steps", 0.0))
            ep_no_progress_sum += float(info.get("no_progress_steps", 0.0))
            ep_dock_idle_sum += float(info.get("dock_idle_steps", 0.0))

            if info["battery_ratio"] <= cfg.robot.low_battery_threshold:
                low_battery_entered = True
            if low_battery_entered and info["on_dock"]:
                returned_on_dock = True

            done = terminated or truncated

        returns.append(ep_ret)
        coverages.append(info["coverage"])

        if len(ep_gate_weights) > 0:
            gate_weight_records.append(
                np.mean(np.asarray(ep_gate_weights), axis=0))
        if len(ep_low_gate) > 0:
            low_battery_gate_records.append(
                np.mean(np.asarray(ep_low_gate), axis=0))
        if len(ep_near_gate) > 0:
            near_obstacle_gate_records.append(
                np.mean(np.asarray(ep_near_gate), axis=0))

        mean_idle_steps.append(ep_idle_sum / max(ep_len, 1))
        mean_no_progress_steps.append(ep_no_progress_sum / max(ep_len, 1))
        mean_dock_idle_steps.append(ep_dock_idle_sum / max(ep_len, 1))

        dr = info["done_reason"]
        if dr in ("collision_static", "collision_dynamic"):
            collision_count += 1
        elif dr == "battery_empty":
            battery_empty_count += 1
        elif dr == "full_clean":
            full_clean_count += 1
        elif dr == "timeout":
            timeout_count += 1

        if returned_on_dock:
            return_success_count += 1

        env.close()

    n = max(eval_episodes, 1)
    metrics = {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_coverage_mean": float(np.mean(coverages)),
        "collision_rate": float(collision_count / n),
        "battery_empty_rate": float(battery_empty_count / n),
        "full_clean_rate": float(full_clean_count / n),
        "timeout_rate": float(timeout_count / n),
        "low_battery_return_success_rate": float(return_success_count / n),
        "mean_idle_steps": float(np.mean(mean_idle_steps)),
        "mean_no_progress_steps": float(np.mean(mean_no_progress_steps)),
        "mean_dock_idle_steps": float(np.mean(mean_dock_idle_steps)),
    }

    if len(gate_weight_records) > 0:
        gate_mean = np.mean(np.asarray(gate_weight_records), axis=0)
        for i, name in enumerate(model.expert_names):
            metrics[f"gate_{name}"] = float(gate_mean[i])

    if len(low_battery_gate_records) > 0:
        gate_mean = np.mean(np.asarray(low_battery_gate_records), axis=0)
        for i, name in enumerate(model.expert_names):
            metrics[f"low_gate_{name}"] = float(gate_mean[i])

    if len(near_obstacle_gate_records) > 0:
        gate_mean = np.mean(np.asarray(near_obstacle_gate_records), axis=0)
        for i, name in enumerate(model.expert_names):
            metrics[f"near_gate_{name}"] = float(gate_mean[i])

    return metrics


# =========================================================
# 主训练
# =========================================================
def train():
    # -------------------------
    # 环境配置
    # -------------------------
    base_cfg = EnvConfig()
    base_cfg.use_dict_observation = True
    base_cfg.training_mode = "baseline_easy"

    seed = base_cfg.seed if base_cfg.seed is not None else 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # PPO 超参数
    # -------------------------
    total_timesteps = 1_200_000
    num_envs = 8
    num_steps = 256

    learning_rate = 1e-4
    gamma = 0.99
    gae_lambda = 0.95

    update_epochs = 4
    minibatch_size = 512

    clip_coef = 0.15
    value_clip_coef = 0.2
    ent_coef = 0.004
    vf_coef = 0.35
    max_grad_norm = 0.5
    target_kl = 0.01

    # -------------------------
    # MoE 正则 / 引导超参数
    # -------------------------
    gate_balance_coef = 0.03
    gate_entropy_coef = 0.003

    # v3 新增：场景弱引导
    gate_guidance_coef = 0.020

    # 场景检测阈值
    avoid_ray_threshold = 0.18
    avoid_map_density_threshold = 0.20
    dock_visible_battery_margin = 0.08

    eval_interval = 10
    eval_episodes = 8

    save_dir = "checkpoints_moeppo"
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # expert 设置
    # -------------------------
    expert_names = ["global_clean", "local_avoid", "return_dock"]

    # -------------------------
    # 初始环境
    # -------------------------
    current_stage = "baseline_easy"
    cfg = build_stage_cfg(base_cfg, current_stage)

    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg, seed, i) for i in range(num_envs)]
    )

    assert isinstance(envs.single_observation_space, gym.spaces.Dict)
    action_dim = envs.single_action_space.shape[0]

    model = MoeActorCritic(
        obs_space=envs.single_observation_space,
        action_dim=action_dim,
        expert_names=expert_names,
        shared_dim=256,
        expert_hidden_dim=128,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    save_json(asdict(cfg), os.path.join(save_dir, "env_config_init.json"))
    save_json(
        {
            "algo": "moe_ppo_baseline_v3_scene_guided",
            "version": "v3",
            "expert_names": expert_names,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
            "num_steps": num_steps,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "update_epochs": update_epochs,
            "minibatch_size": minibatch_size,
            "clip_coef": clip_coef,
            "value_clip_coef": value_clip_coef,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "target_kl": target_kl,
            "gate_balance_coef": gate_balance_coef,
            "gate_entropy_coef": gate_entropy_coef,
            "gate_guidance_coef": gate_guidance_coef,
            "avoid_ray_threshold": avoid_ray_threshold,
            "avoid_map_density_threshold": avoid_map_density_threshold,
            "dock_visible_battery_margin": dock_visible_battery_margin,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
        },
        os.path.join(save_dir, "train_config.json"),
    )

    obs, info = envs.reset(seed=seed)
    current_info = info

    # -------------------------
    # rollout buffer
    # -------------------------
    obs_buf = {
        k: np.zeros((num_steps, num_envs) +
                    envs.single_observation_space[k].shape, dtype=np.float32)
        for k in envs.single_observation_space.spaces.keys()
    }

    actions_buf = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
    logprobs_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
    rewards_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
    dones_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
    values_buf = np.zeros((num_steps, num_envs), dtype=np.float32)

    # gate / guidance rollout 额外缓存
    gates_buf = np.zeros(
        (num_steps, num_envs, len(expert_names)), dtype=np.float32)
    guide_targets_buf = np.zeros(
        (num_steps, num_envs, len(expert_names)), dtype=np.float32)
    low_mask_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
    near_mask_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
    dock_visible_mask_buf = np.zeros((num_steps, num_envs), dtype=np.float32)

    global_step = 0
    num_updates = total_timesteps // (num_envs * num_steps)

    # -------------------------
    # episode tracking
    # -------------------------
    running_ep_return = np.zeros(num_envs, dtype=np.float32)
    running_low_battery_entered = np.zeros(num_envs, dtype=bool)
    running_returned_on_dock = np.zeros(num_envs, dtype=bool)

    ep_return_hist = []
    ep_coverage_hist = []
    ep_collision_hist = []
    ep_battery_empty_hist = []
    ep_full_clean_hist = []
    ep_timeout_hist = []
    ep_return_success_hist = []
    ep_idle_hist = []
    ep_no_progress_hist = []
    ep_dock_idle_hist = []

    # -------------------------
    # best model tracking
    # -------------------------
    best_return = -1e18
    best_coverage = -1e18
    best_safe_score = -1e18
    best_return_success = -1e18
    best_anti_stall_score = -1e18

    start_time = time.time()

    for update in range(1, num_updates + 1):
        # curriculum stage switch
        wanted_stage = stage_name_by_update(update, num_updates)
        if wanted_stage != current_stage:
            current_stage = wanted_stage
            cfg = build_stage_cfg(base_cfg, current_stage)

            envs.close()
            envs = gym.vector.SyncVectorEnv(
                [make_env(cfg, seed + update * 1000, i)
                 for i in range(num_envs)]
            )
            obs, info = envs.reset(seed=seed + update * 1000)
            current_info = info

            print(f"[Stage Switch] training_mode -> {current_stage}")

            obs_buf = {
                k: np.zeros((num_steps, num_envs) +
                            envs.single_observation_space[k].shape, dtype=np.float32)
                for k in envs.single_observation_space.spaces.keys()
            }

        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = learning_rate * frac

        for step in range(num_steps):
            global_step += num_envs

            # 存当前 obs
            for k in obs_buf.keys():
                obs_buf[k][step] = obs[k]

            # 基于“当前状态”做弱引导目标构造
            guide_targets, low_mask, near_mask, dock_visible_mask = derive_guidance_targets(
                obs=obs,
                info=current_info,
                cfg=cfg,
                expert_names=expert_names,
                avoid_ray_threshold=avoid_ray_threshold,
                avoid_map_density_threshold=avoid_map_density_threshold,
                dock_visible_battery_margin=dock_visible_battery_margin,
            )
            guide_targets_buf[step] = guide_targets
            low_mask_buf[step] = low_mask
            near_mask_buf[step] = near_mask
            dock_visible_mask_buf[step] = dock_visible_mask

            obs_t = obs_to_tensor(obs, device)

            with torch.no_grad():
                action_t, logprob_t, _, value_t, aux = model.get_action_value(
                    obs_t)

            action = action_t.cpu().numpy()

            next_obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.logical_or(terminated, truncated)

            actions_buf[step] = action
            logprobs_buf[step] = logprob_t.cpu().numpy()
            rewards_buf[step] = reward
            dones_buf[step] = done.astype(np.float32)
            values_buf[step] = value_t.cpu().numpy()
            gates_buf[step] = aux["gate_weights"].cpu().numpy()

            running_ep_return += reward

            if "battery_ratio" in infos:
                battery_ratio_arr = np.asarray(infos["battery_ratio"])
                running_low_battery_entered = np.logical_or(
                    running_low_battery_entered,
                    battery_ratio_arr <= cfg.robot.low_battery_threshold,
                )

            if "on_dock" in infos:
                on_dock_arr = np.asarray(infos["on_dock"]).astype(bool)
                running_returned_on_dock = np.logical_or(
                    running_returned_on_dock,
                    np.logical_and(running_low_battery_entered, on_dock_arr),
                )

            final_info_list = extract_final_info_list(infos, num_envs)

            for i in range(num_envs):
                if not done[i]:
                    continue

                finfo = final_info_list[i]

                if finfo is None:
                    finfo = {}
                    for key, value in infos.items():
                        if key in ("final_info", "_final_info", "final_observation", "_final_observation"):
                            continue
                        try:
                            if isinstance(value, np.ndarray) and len(value) == num_envs:
                                finfo[key] = value[i].item() if np.ndim(
                                    value[i]) == 0 else value[i]
                        except Exception:
                            pass

                ep_return_hist.append(float(running_ep_return[i]))
                ep_coverage_hist.append(float(finfo.get("coverage", 0.0)))

                dr = finfo.get("done_reason", "unknown")
                ep_collision_hist.append(1.0 if dr in (
                    "collision_static", "collision_dynamic") else 0.0)
                ep_battery_empty_hist.append(
                    1.0 if dr == "battery_empty" else 0.0)
                ep_full_clean_hist.append(1.0 if dr == "full_clean" else 0.0)
                ep_timeout_hist.append(1.0 if dr == "timeout" else 0.0)
                ep_return_success_hist.append(
                    1.0 if running_returned_on_dock[i] else 0.0)

                ep_idle_hist.append(float(finfo.get("idle_steps", 0.0)))
                ep_no_progress_hist.append(
                    float(finfo.get("no_progress_steps", 0.0)))
                ep_dock_idle_hist.append(
                    float(finfo.get("dock_idle_steps", 0.0)))

                running_ep_return[i] = 0.0
                running_low_battery_entered[i] = False
                running_returned_on_dock[i] = False

            obs = next_obs
            current_info = infos

        # -------------------------
        # GAE
        # -------------------------
        with torch.no_grad():
            next_obs_t = obs_to_tensor(obs, device)
            _, next_value_t, _ = model.get_dist_and_value(next_obs_t)
            next_value = next_value_t.cpu().numpy()

        advantages = np.zeros_like(rewards_buf)
        lastgaelam = np.zeros(num_envs, dtype=np.float32)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - dones_buf[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]

            delta = rewards_buf[t] + gamma * nextvalues * \
                nextnonterminal - values_buf[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values_buf

        # -------------------------
        # flatten batch
        # -------------------------
        b_obs = {
            k: torch.as_tensor(
                v.reshape((-1,) + v.shape[2:]), dtype=torch.float32, device=device)
            for k, v in obs_buf.items()
        }
        b_actions = torch.as_tensor(
            actions_buf.reshape(-1, action_dim), dtype=torch.float32, device=device)
        b_logprobs = torch.as_tensor(
            logprobs_buf.reshape(-1), dtype=torch.float32, device=device)
        b_advantages = torch.as_tensor(
            advantages.reshape(-1), dtype=torch.float32, device=device)
        b_returns = torch.as_tensor(
            returns.reshape(-1), dtype=torch.float32, device=device)
        b_values = torch.as_tensor(
            values_buf.reshape(-1), dtype=torch.float32, device=device)
        b_guide_targets = torch.as_tensor(
            guide_targets_buf.reshape(-1, len(expert_names)), dtype=torch.float32, device=device)

        b_advantages = (b_advantages - b_advantages.mean()) / \
            (b_advantages.std() + 1e-8)

        batch_size = num_envs * num_steps
        batch_inds = np.arange(batch_size)

        last_pg_loss = 0.0
        last_v_loss = 0.0
        last_entropy = 0.0
        last_gate_balance = 0.0
        last_gate_entropy = 0.0
        last_gate_guidance = 0.0
        all_clipfracs = []
        all_approx_kl = []

        # rollout 统计日志
        flat_gates = gates_buf.reshape(-1, len(expert_names))
        flat_low_mask = low_mask_buf.reshape(-1).astype(bool)
        flat_near_mask = near_mask_buf.reshape(-1).astype(bool)

        mean_gate_weights = flat_gates.mean(axis=0)
        low_gate_weights = flat_gates[flat_low_mask].mean(
            axis=0) if np.any(flat_low_mask) else None
        near_gate_weights = flat_gates[flat_near_mask].mean(
            axis=0) if np.any(flat_near_mask) else None

        for epoch in range(update_epochs):
            np.random.shuffle(batch_inds)
            epoch_approx_kl = []

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = batch_inds[start:end]

                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                _, newlogprob, entropy, newvalue, aux = model.get_action_value(
                    mb_obs, b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    epoch_approx_kl.append(float(approx_kl.cpu().item()))
                    all_approx_kl.append(float(approx_kl.cpu().item()))

                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                    all_clipfracs.append(float(clipfrac.cpu().item()))

                mb_adv = b_advantages[mb_inds]

                # PPO policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * \
                    torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -value_clip_coef,
                    value_clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * \
                    torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # gate anti-collapse
                gate_balance_loss, gate_entropy, _ = compute_gate_regularization(
                    aux["gate_weights"],
                    num_experts=len(expert_names),
                )

                # v3: gate 场景弱引导
                gate_guidance_loss = compute_gate_guidance_loss(
                    aux["gate_weights"],
                    b_guide_targets[mb_inds],
                )

                entropy_loss = entropy.mean()

                loss = (
                    pg_loss
                    - ent_coef * entropy_loss
                    + vf_coef * v_loss
                    + gate_balance_coef * gate_balance_loss
                    - gate_entropy_coef * gate_entropy
                    + gate_guidance_coef * gate_guidance_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                last_pg_loss = float(pg_loss.item())
                last_v_loss = float(v_loss.item())
                last_entropy = float(entropy_loss.item())
                last_gate_balance = float(gate_balance_loss.item())
                last_gate_entropy = float(gate_entropy.item())
                last_gate_guidance = float(gate_guidance_loss.item())

            if len(epoch_approx_kl) > 0 and np.mean(epoch_approx_kl) > target_kl:
                break

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        ev = explained_variance(y_pred, y_true)

        # -------------------------
        # 日志
        # -------------------------
        sps = int(global_step / (time.time() - start_time + 1e-8))

        recent_ret = safe_mean(ep_return_hist[-50:])
        recent_cov = safe_mean(ep_coverage_hist[-50:])
        recent_collision = safe_mean(ep_collision_hist[-50:])
        recent_battery_empty = safe_mean(ep_battery_empty_hist[-50:])
        recent_full_clean = safe_mean(ep_full_clean_hist[-50:])
        recent_return_success = safe_mean(ep_return_success_hist[-50:])
        recent_idle = safe_mean(ep_idle_hist[-50:])
        recent_no_progress = safe_mean(ep_no_progress_hist[-50:])
        recent_dock_idle = safe_mean(ep_dock_idle_hist[-50:])

        gate_log_str = " ".join([
            f"gate_{expert_names[i]}={mean_gate_weights[i]:.3f}"
            for i in range(len(expert_names))
        ])

        if low_gate_weights is not None:
            low_gate_log_str = " ".join([
                f"low_gate_{expert_names[i]}={low_gate_weights[i]:.3f}"
                for i in range(len(expert_names))
            ])
        else:
            low_gate_log_str = "low_gate_NA"

        if near_gate_weights is not None:
            near_gate_log_str = " ".join([
                f"near_gate_{expert_names[i]}={near_gate_weights[i]:.3f}"
                for i in range(len(expert_names))
            ])
        else:
            near_gate_log_str = "near_gate_NA"

        print(
            f"[Update {update}/{num_updates}] "
            f"mode={current_stage} "
            f"step={global_step} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"SPS={sps} "
            f"pg={last_pg_loss:.4f} "
            f"v={last_v_loss:.4f} "
            f"ent={last_entropy:.4f} "
            f"gate_bal={last_gate_balance:.6f} "
            f"gate_ent={last_gate_entropy:.4f} "
            f"gate_guid={last_gate_guidance:.4f} "
            f"kl={safe_mean(all_approx_kl):.6f} "
            f"clipfrac={safe_mean(all_clipfracs):.4f} "
            f"ev={ev:.4f} "
            f"ep_ret={recent_ret:.3f} "
            f"cov={recent_cov:.3f} "
            f"coll={recent_collision:.3f} "
            f"battery_empty={recent_battery_empty:.3f} "
            f"full_clean={recent_full_clean:.3f} "
            f"return_success={recent_return_success:.3f} "
            f"idle={recent_idle:.2f} "
            f"noprog={recent_no_progress:.2f} "
            f"dock_idle={recent_dock_idle:.2f} "
            f"{gate_log_str} "
            f"{low_gate_log_str} "
            f"{near_gate_log_str}"
        )

        # -------------------------
        # 评估与保存
        # -------------------------
        if update % eval_interval == 0 or update == num_updates:
            eval_cfg = build_stage_cfg(base_cfg, current_stage)

            metrics = evaluate_policy(
                model=model,
                device=device,
                cfg=eval_cfg,
                eval_episodes=eval_episodes,
                render=False,
            )

            eval_gate_str = " ".join([
                f"gate_{name}={metrics.get(f'gate_{name}', 0.0):.3f}"
                for name in expert_names
            ])

            eval_low_gate_str = " ".join([
                f"low_gate_{name}={metrics.get(f'low_gate_{name}', 0.0):.3f}"
                for name in expert_names
            ]) if any(f"low_gate_{name}" in metrics for name in expert_names) else "low_gate_NA"

            eval_near_gate_str = " ".join([
                f"near_gate_{name}={metrics.get(f'near_gate_{name}', 0.0):.3f}"
                for name in expert_names
            ]) if any(f"near_gate_{name}" in metrics for name in expert_names) else "near_gate_NA"

            print(
                f"[Eval] "
                f"mode={current_stage} "
                f"ret={metrics['eval_return_mean']:.3f}±{metrics['eval_return_std']:.3f} "
                f"cov={metrics['eval_coverage_mean']:.3f} "
                f"coll={metrics['collision_rate']:.3f} "
                f"battery_empty={metrics['battery_empty_rate']:.3f} "
                f"full_clean={metrics['full_clean_rate']:.3f} "
                f"timeout={metrics['timeout_rate']:.3f} "
                f"return_success={metrics['low_battery_return_success_rate']:.3f} "
                f"idle={metrics['mean_idle_steps']:.2f} "
                f"noprog={metrics['mean_no_progress_steps']:.2f} "
                f"dock_idle={metrics['mean_dock_idle_steps']:.2f} "
                f"{eval_gate_str} "
                f"{eval_low_gate_str} "
                f"{eval_near_gate_str}"
            )

            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": asdict(eval_cfg),
                "metrics": metrics,
                "global_step": global_step,
                "update": update,
                "training_mode": current_stage,
                "expert_names": expert_names,
                "version": "v3",
            }

            torch.save(ckpt, os.path.join(save_dir, "latest_model.pt"))

            if metrics["eval_return_mean"] > best_return:
                best_return = metrics["eval_return_mean"]
                torch.save(ckpt, os.path.join(
                    save_dir, "best_return_model.pt"))

            if metrics["eval_coverage_mean"] > best_coverage:
                best_coverage = metrics["eval_coverage_mean"]
                torch.save(ckpt, os.path.join(
                    save_dir, "best_coverage_model.pt"))

            safe_score = (
                metrics["eval_coverage_mean"]
                - 0.5 * metrics["collision_rate"]
                - 0.3 * metrics["battery_empty_rate"]
            )
            if safe_score > best_safe_score:
                best_safe_score = safe_score
                torch.save(ckpt, os.path.join(save_dir, "best_safe_model.pt"))

            if metrics["low_battery_return_success_rate"] > best_return_success:
                best_return_success = metrics["low_battery_return_success_rate"]
                torch.save(ckpt, os.path.join(
                    save_dir, "best_return_success_model.pt"))

            anti_stall_score = (
                metrics["eval_coverage_mean"]
                + 0.25 * metrics["low_battery_return_success_rate"]
                - 0.02 * metrics["mean_idle_steps"]
                - 0.02 * metrics["mean_no_progress_steps"]
                - 0.03 * metrics["mean_dock_idle_steps"]
                - 0.40 * metrics["collision_rate"]
                - 0.20 * metrics["battery_empty_rate"]
            )
            if anti_stall_score > best_anti_stall_score:
                best_anti_stall_score = anti_stall_score
                torch.save(ckpt, os.path.join(
                    save_dir, "best_anti_stall_model.pt"))

    envs.close()
    print("训练结束，模型已保存到:", save_dir)


if __name__ == "__main__":
    train()
