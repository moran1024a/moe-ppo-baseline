from cleaning_robot_env import CleaningRobotEnv
from env_cfg import EnvConfig
import pygame
import gymnasium as gym
import torch.nn as nn
import torch
import numpy as np
import os
import time
import argparse
from typing import Optional, Dict, Any

# ---------------------------------------------------------
# 关键：在导入 pygame / 环境之前先设 SDL
# WSL 下通常 x11 更稳；如果你是 WSLg，也可改成 wayland 试
# ---------------------------------------------------------
os.environ.setdefault("SDL_VIDEODRIVER", "x11")


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
EPS = 1e-6


def stack_local_maps(obs_dict: dict) -> torch.Tensor:
    maps = [obs_dict["local_clean_map"], obs_dict["local_obstacle_map"]]
    if "local_dock_map" in obs_dict:
        maps.append(obs_dict["local_dock_map"])
    return torch.stack(maps, dim=1)


def build_cfg_from_ckpt(ckpt_cfg: Optional[Dict[str, Any]]) -> EnvConfig:
    cfg = EnvConfig()
    if ckpt_cfg is None:
        return cfg

    if "dt" in ckpt_cfg:
        cfg.dt = ckpt_cfg["dt"]
    if "max_steps" in ckpt_cfg:
        cfg.max_steps = ckpt_cfg["max_steps"]
    if "seed" in ckpt_cfg:
        cfg.seed = ckpt_cfg["seed"]
    if "use_dict_observation" in ckpt_cfg:
        cfg.use_dict_observation = ckpt_cfg["use_dict_observation"]
    if "training_mode" in ckpt_cfg:
        cfg.training_mode = ckpt_cfg["training_mode"]

    for sub_name in ["robot", "map_cfg", "sensor", "dynamic_obstacle", "reward", "render", "curriculum"]:
        if sub_name in ckpt_cfg and isinstance(ckpt_cfg[sub_name], dict):
            sub_cfg = getattr(cfg, sub_name)
            for k, v in ckpt_cfg[sub_name].items():
                if hasattr(sub_cfg, k):
                    setattr(sub_cfg, k, v)

    return cfg


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
        self.net = nn.Sequential(
            nn.Linear(num_rays * ray_dim, 128),
            nn.Tanh(),
            nn.Linear(128, out_dim),
            nn.Tanh(),
        )

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        return self.net(rays.reshape(rays.shape[0], -1))


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
            obs_space, gym.spaces.Dict), "review 要求 dict observation"

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

    def act_deterministic(self, obs_dict: dict):
        feat = self.encode(obs_dict)
        mean = self.actor_mean(feat)
        return torch.tanh(mean)


def warmup_window(env: CleaningRobotEnv, frames: int = 5, delay: float = 0.03) -> None:
    """
    WSL 下预热窗口，避免只出现任务栏图标但窗口不激活。
    """
    for _ in range(frames):
        env.render()
        pygame.event.pump()
        time.sleep(delay)


@torch.no_grad()
def run_review(
    ckpt_path: str,
    episodes: int = 10,
    render: bool = False,
    training_mode: Optional[str] = None,
    log_interval: int = 100,
    seed: Optional[int] = None,
):
    print(
        f"[Review] start, model_path={ckpt_path}, render={render}, episodes={episodes}, seed={seed}",
        flush=True
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型文件: {ckpt_path}")

    # ---------------------------------------------------------
    # 回测强制 CPU，减少 WSL 下 GUI / CUDA 干扰
    # ---------------------------------------------------------
    device = torch.device("cpu")
    print(f"[Review] device={device}", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = build_cfg_from_ckpt(ckpt.get("cfg", None))
    cfg.use_dict_observation = True

    if training_mode is not None:
        cfg.training_mode = training_mode

    review_seed = seed if seed is not None else (
        cfg.seed if cfg.seed is not None else 45)

    print(
        f"[Review] cfg loaded, training_mode={cfg.training_mode}, "
        f"use_local_dock_map={cfg.sensor.use_local_dock_map}, "
        f"max_steps={cfg.max_steps}, review_seed={review_seed}",
        flush=True
    )

    env = CleaningRobotEnv(cfg=cfg, render_mode="human" if render else None)
    model = ActorCritic(env.observation_space,
                        env.action_space.shape[0]).to(device)

    if "model_state_dict" not in ckpt:
        raise KeyError("checkpoint 中缺少 model_state_dict")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    episode_returns = []
    episode_coverages = []
    episode_steps = []
    final_batteries = []

    episode_idle_last = []
    episode_no_progress_last = []
    episode_dock_idle_last = []

    done_reason_count = {
        "collision_static": 0,
        "collision_dynamic": 0,
        "battery_empty": 0,
        "full_clean": 0,
        "timeout": 0,
        "unknown": 0,
    }

    collision_count = 0
    battery_empty_count = 0
    full_clean_count = 0
    timeout_count = 0
    return_success_count = 0

    for ep in range(episodes):
        ep_seed = review_seed + 5000 + ep
        print(
            f"[Review] episode {ep + 1}/{episodes} reset, seed={ep_seed} ...", flush=True)
        obs, info = env.reset(seed=ep_seed)

        if render:
            warmup_window(env, frames=6, delay=0.03)

        done = False
        ep_return = 0.0
        step_count = 0
        low_battery_entered = False
        ever_returned_on_dock = False
        max_coverage = float(info.get("coverage", 0.0))

        last_idle_steps = 0.0
        last_no_progress_steps = 0.0
        last_dock_idle_steps = 0.0

        print(
            f"[Episode {ep + 1}] reset_done "
            f"coverage={info.get('coverage', 0.0):.3f} "
            f"battery={info.get('battery', 0.0):.1f}",
            flush=True
        )

        while not done:
            obs_t = {
                k: torch.as_tensor(np.expand_dims(
                    v, 0), dtype=torch.float32, device=device)
                for k, v in obs.items()
            }
            action = model.act_deterministic(obs_t).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            step_count += 1
            max_coverage = max(max_coverage, float(info["coverage"]))

            last_idle_steps = float(info.get("idle_steps", 0.0))
            last_no_progress_steps = float(info.get("no_progress_steps", 0.0))
            last_dock_idle_steps = float(info.get("dock_idle_steps", 0.0))

            if info["battery_ratio"] <= cfg.robot.low_battery_threshold:
                low_battery_entered = True
            if low_battery_entered and info["on_dock"]:
                ever_returned_on_dock = True

            if render:
                pygame.event.pump()

            if step_count % log_interval == 0:
                print(
                    f"[Episode {ep + 1}] step={step_count} "
                    f"ret={ep_return:.3f} "
                    f"coverage={info['coverage']:.3f} "
                    f"battery={info['battery']:.1f} "
                    f"idle={last_idle_steps:.0f} "
                    f"noprog={last_no_progress_steps:.0f} "
                    f"dock_idle={last_dock_idle_steps:.0f} "
                    f"done_reason={info.get('done_reason', None)}",
                    flush=True
                )

            done = terminated or truncated

        dr = info.get("done_reason", None)
        if dr in done_reason_count:
            done_reason_count[dr] += 1
        else:
            done_reason_count["unknown"] += 1

        if dr in ("collision_static", "collision_dynamic"):
            collision_count += 1
        if dr == "battery_empty":
            battery_empty_count += 1
        if dr == "full_clean":
            full_clean_count += 1
        if dr == "timeout":
            timeout_count += 1
        if ever_returned_on_dock:
            return_success_count += 1

        episode_returns.append(ep_return)
        episode_coverages.append(float(info["coverage"]))
        episode_steps.append(step_count)
        final_batteries.append(float(info["battery"]))

        episode_idle_last.append(last_idle_steps)
        episode_no_progress_last.append(last_no_progress_steps)
        episode_dock_idle_last.append(last_dock_idle_steps)

        print(
            f"[Episode {ep + 1}/{episodes}] "
            f"seed={ep_seed} "
            f"return={ep_return:.3f} "
            f"final_coverage={info['coverage']:.3f} "
            f"max_coverage={max_coverage:.3f} "
            f"steps={step_count} "
            f"battery={info['battery']:.1f} "
            f"idle={last_idle_steps:.0f} "
            f"noprog={last_no_progress_steps:.0f} "
            f"dock_idle={last_dock_idle_steps:.0f} "
            f"done_reason={dr} "
            f"return_success={ever_returned_on_dock}",
            flush=True
        )

    env.close()

    n = max(episodes, 1)

    print("\n" + "=" * 72, flush=True)
    print("Review Summary", flush=True)
    print("=" * 72, flush=True)
    print(f"model_path: {ckpt_path}", flush=True)
    print(f"episodes: {episodes}", flush=True)
    print(f"training_mode: {cfg.training_mode}", flush=True)
    print(f"base_seed: {review_seed}", flush=True)
    print(f"avg_return: {np.mean(episode_returns):.4f}", flush=True)
    print(f"std_return: {np.std(episode_returns):.4f}", flush=True)
    print(f"avg_final_coverage: {np.mean(episode_coverages):.4f}", flush=True)
    print(f"avg_steps: {np.mean(episode_steps):.2f}", flush=True)
    print(f"avg_final_battery: {np.mean(final_batteries):.2f}", flush=True)
    print(f"avg_last_idle_steps: {np.mean(episode_idle_last):.2f}", flush=True)
    print(
        f"avg_last_no_progress_steps: {np.mean(episode_no_progress_last):.2f}", flush=True)
    print(
        f"avg_last_dock_idle_steps: {np.mean(episode_dock_idle_last):.2f}", flush=True)
    print(f"collision_rate: {collision_count / n:.4f}", flush=True)
    print(f"battery_empty_rate: {battery_empty_count / n:.4f}", flush=True)
    print(f"full_clean_rate: {full_clean_count / n:.4f}", flush=True)
    print(f"timeout_rate: {timeout_count / n:.4f}", flush=True)
    print(
        f"low_battery_return_success_rate: {return_success_count / n:.4f}", flush=True)
    print("done_reason_count:", done_reason_count, flush=True)

    if "metrics" in ckpt:
        print("\ncheckpoint saved metrics:", ckpt["metrics"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_return_model.pt",
        help="模型路径",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="回测 episode 数",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="是否开启可视化",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default=None,
        choices=["baseline_easy", "baseline_mid", "baseline_full"],
        help="可选覆盖环境难度；默认使用 checkpoint 内保存的配置",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="每多少步打印一次过程日志",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="回测随机种子；若不传则优先使用 checkpoint 中的 seed，否则默认 45",
    )

    args = parser.parse_args()

    run_review(
        ckpt_path=args.model_path,
        episodes=args.episodes,
        render=args.render,
        training_mode=args.training_mode,
        log_interval=args.log_interval,
        seed=args.seed,
    )
