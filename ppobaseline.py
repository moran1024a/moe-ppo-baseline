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
        tanh 后的精确 entropy 不直接写，这里用 unsquashed normal entropy 做监控近似
        """
        return self.normal.entropy().sum(dim=-1)

    def mode(self):
        return torch.tanh(self.mean)


# =========================================================
# 网络
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


class ActorCritic(nn.Module):
    def __init__(self, obs_space: gym.Space, action_dim: int):
        super().__init__()
        assert isinstance(
            obs_space, gym.spaces.Dict), "当前版本要求 dict observation"

        state_dim = obs_space["state"].shape[0]
        num_rays = obs_space["rays"].shape[0]
        ray_dim = obs_space["rays"].shape[1]
        local_map_size = obs_space["local_clean_map"].shape[0]
        in_channels = 2 + (1 if "local_dock_map" in obs_space.spaces else 0)

        self.state_encoder = StateEncoder(state_dim, 128)
        self.ray_encoder = RayEncoder(num_rays, ray_dim, 128)
        self.map_encoder = LocalMapEncoder(in_channels, local_map_size, 128)

        self.backbone = nn.Sequential(
            nn.Linear(384, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * -1.1)
        self.critic = nn.Linear(256, 1)

    def encode(self, obs_dict: dict) -> torch.Tensor:
        state_feat = self.state_encoder(obs_dict["state"])
        ray_feat = self.ray_encoder(obs_dict["rays"])
        map_feat = self.map_encoder(stack_local_maps(obs_dict))
        x = torch.cat([state_feat, ray_feat, map_feat], dim=-1)
        return self.backbone(x)

    def get_dist_and_value(self, obs_dict: dict):
        feat = self.encode(obs_dict)
        mean = self.actor_mean(feat)
        log_std = self.actor_log_std.unsqueeze(0).expand_as(mean)
        dist = SquashedNormal(mean, log_std)
        value = self.critic(feat).squeeze(-1)
        return dist, value

    def get_action_value(self, obs_dict: dict, action=None):
        dist, value = self.get_dist_and_value(obs_dict)
        if action is None:
            action, log_prob = dist.sample()
        else:
            action = torch.clamp(action, -1.0 + EPS, 1.0 - EPS)
            log_prob = dist.log_prob(action)
        entropy = dist.entropy_proxy()
        return action, log_prob, entropy, value

    def act_deterministic(self, obs_dict: dict):
        dist, _ = self.get_dist_and_value(obs_dict)
        return dist.mode()


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
# 训练阶段切换
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
# 评估
# =========================================================
@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
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

        while not done:
            obs_t = {
                k: torch.as_tensor(np.expand_dims(
                    v, 0), dtype=torch.float32, device=device)
                for k, v in obs.items()
            }
            action = model.act_deterministic(obs_t).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
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
    return {
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

    eval_interval = 10
    eval_episodes = 8

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

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

    model = ActorCritic(envs.single_observation_space, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    save_json(asdict(cfg), os.path.join(save_dir, "env_config_init.json"))
    save_json(
        {
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
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
        },
        os.path.join(save_dir, "train_config.json"),
    )

    obs, info = envs.reset(seed=seed)

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

            for k in obs_buf.keys():
                obs_buf[k][step] = obs[k]

            obs_t = obs_to_tensor(obs, device)

            with torch.no_grad():
                action_t, logprob_t, _, value_t = model.get_action_value(obs_t)

            action = action_t.cpu().numpy()

            next_obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.logical_or(terminated, truncated)

            actions_buf[step] = action
            logprobs_buf[step] = logprob_t.cpu().numpy()
            rewards_buf[step] = reward
            dones_buf[step] = done.astype(np.float32)
            values_buf[step] = value_t.cpu().numpy()

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

        # -------------------------
        # GAE
        # -------------------------
        with torch.no_grad():
            next_obs_t = obs_to_tensor(obs, device)
            _, next_value_t = model.get_dist_and_value(next_obs_t)
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

        # flatten batch
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

        b_advantages = (b_advantages - b_advantages.mean()) / \
            (b_advantages.std() + 1e-8)

        batch_size = num_envs * num_steps
        batch_inds = np.arange(batch_size)

        last_pg_loss = 0.0
        last_v_loss = 0.0
        last_entropy = 0.0
        all_clipfracs = []
        all_approx_kl = []

        for epoch in range(update_epochs):
            np.random.shuffle(batch_inds)
            epoch_approx_kl = []

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = batch_inds[start:end]

                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                _, newlogprob, entropy, newvalue = model.get_action_value(
                    mb_obs, b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    epoch_approx_kl.append(float(approx_kl.cpu().item()))
                    all_approx_kl.append(float(approx_kl.cpu().item()))

                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                    all_clipfracs.append(float(clipfrac.cpu().item()))

                mb_adv = b_advantages[mb_inds]

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * \
                    torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                last_pg_loss = float(pg_loss.item())
                last_v_loss = float(v_loss.item())
                last_entropy = float(entropy_loss.item())

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

        print(
            f"[Update {update}/{num_updates}] "
            f"mode={current_stage} "
            f"step={global_step} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"SPS={sps} "
            f"pg={last_pg_loss:.4f} "
            f"v={last_v_loss:.4f} "
            f"ent={last_entropy:.4f} "
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
            f"dock_idle={recent_dock_idle:.2f}"
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
                f"dock_idle={metrics['mean_dock_idle_steps']:.2f}"
            )

            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": asdict(eval_cfg),
                "metrics": metrics,
                "global_step": global_step,
                "update": update,
                "training_mode": current_stage,
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
