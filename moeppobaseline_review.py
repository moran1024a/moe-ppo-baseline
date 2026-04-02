import os
import time
import argparse

import numpy as np
import torch

from env_cfg import EnvConfig
from cleaning_robot_env import CleaningRobotEnv

# 直接复用训练脚本中的模型与配置函数
from moeppobaseline import MoeActorCritic, build_stage_cfg


# =========================================================
# 基础工具
# =========================================================
def obs_to_tensor_single(obs_dict: dict, device: torch.device) -> dict:
    """
    单环境 obs -> 带 batch 维度 tensor
    """
    out = {}
    for k, v in obs_dict.items():
        out[k] = torch.as_tensor(
            np.expand_dims(v, axis=0),
            dtype=torch.float32,
            device=device,
        )
    return out


def format_gate_info(expert_names, gate_weights: np.ndarray) -> str:
    return " ".join([
        f"{expert_names[i]}={gate_weights[i]:.3f}"
        for i in range(len(expert_names))
    ])


def find_dominant_expert(expert_names, gate_weights: np.ndarray):
    idx = int(np.argmax(gate_weights))
    return idx, expert_names[idx], float(gate_weights[idx])


def resolve_checkpoint_path(path: str) -> str:
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        cand = os.path.join(path, "latest_model.pt")
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(f"找不到 checkpoint: {path}")


def get_info_scalar(info: dict, key: str, default):
    if info is None:
        return default
    return info.get(key, default)


# =========================================================
# 与训练脚本一致的场景判断逻辑
# =========================================================
def derive_single_step_scene_flags(
    obs: dict,
    info: dict,
    cfg: EnvConfig,
    avoid_ray_threshold: float = 0.18,
    avoid_map_density_threshold: float = 0.20,
    dock_visible_battery_margin: float = 0.08,
):
    """
    与训练脚本里的 derive_guidance_targets 保持语义一致，
    这里只做单步场景标记，方便 review 统计。
    """
    battery_ratio = float(get_info_scalar(info, "battery_ratio", 1.0))
    low_battery = battery_ratio <= float(cfg.robot.low_battery_threshold)

    if "local_dock_map" in obs:
        dock_visible = bool(np.sum(obs["local_dock_map"]) > 1e-6)
    else:
        dock_visible = False

    near_low_battery = battery_ratio <= float(
        cfg.robot.low_battery_threshold + dock_visible_battery_margin)
    dock_guided = low_battery or (near_low_battery and dock_visible)

    rays = obs["rays"]
    ray_min = float(np.min(rays[..., 0]))

    obstacle_density = float(np.mean(obs["local_obstacle_map"]))
    near_obstacle = (ray_min < avoid_ray_threshold) or (
        obstacle_density > avoid_map_density_threshold)

    return {
        "battery_ratio": battery_ratio,
        "low_battery": bool(low_battery),
        "dock_visible": bool(dock_visible),
        "dock_guided": bool(dock_guided),
        "ray_min": ray_min,
        "obstacle_density": obstacle_density,
        "near_obstacle": bool(near_obstacle),
    }


# =========================================================
# 加载模型
# =========================================================
def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    saved_cfg_dict = ckpt.get("cfg", None)
    if saved_cfg_dict is None:
        raise RuntimeError("checkpoint 中未找到 cfg")

    base_cfg = EnvConfig()

    # 顶层字段
    for k, v in saved_cfg_dict.items():
        if hasattr(base_cfg, k) and not isinstance(v, dict):
            setattr(base_cfg, k, v)

    # 嵌套配置对象
    nested_keys = [
        "robot",
        "map_cfg",
        "sensor",
        "dynamic_obstacle",
        "reward",
        "render",
        "curriculum",
    ]
    for key in nested_keys:
        if key in saved_cfg_dict and hasattr(base_cfg, key):
            obj = getattr(base_cfg, key)
            if isinstance(saved_cfg_dict[key], dict):
                for kk, vv in saved_cfg_dict[key].items():
                    if hasattr(obj, kk):
                        setattr(obj, kk, vv)

    env = CleaningRobotEnv(cfg=base_cfg, render_mode=None)
    obs_space = env.observation_space
    action_space = env.action_space
    env.close()

    expert_names = ckpt.get(
        "expert_names",
        ["global_clean", "local_avoid", "return_dock"]
    )

    model = MoeActorCritic(
        obs_space=obs_space,
        action_dim=action_space.shape[0],
        expert_names=expert_names,
        shared_dim=256,
        expert_hidden_dim=128,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, base_cfg, ckpt


# =========================================================
# 单回合评测
# =========================================================
@torch.no_grad()
def run_one_episode(
    model: MoeActorCritic,
    cfg: EnvConfig,
    device: torch.device,
    seed: int,
    render: bool = True,
    sleep_time: float = 0.0,
    max_print_steps: int = 100000,
):
    env = CleaningRobotEnv(cfg=cfg, render_mode="human" if render else None)
    obs, info = env.reset(seed=seed)

    done = False
    step_idx = 0
    ep_return = 0.0

    low_battery_entered = False
    returned_on_dock = False

    gate_history = []
    dominant_history = []

    low_battery_gate_history = []
    low_battery_dominant_history = []

    near_obstacle_gate_history = []
    near_obstacle_dominant_history = []

    prev_dom_idx = None
    dominant_switches = 0

    while not done:
        scene_flags = derive_single_step_scene_flags(obs, info, cfg)

        obs_t = obs_to_tensor_single(obs, device)

        action_t, aux = model.act_deterministic(obs_t)
        _, value_t, _ = model.get_dist_and_value(obs_t)

        action = action_t.cpu().numpy()[0]
        value = float(value_t.cpu().numpy()[0])

        gate_weights = aux["gate_weights"].cpu().numpy()[0]
        dom_idx, dom_name, dom_weight = find_dominant_expert(
            model.expert_names, gate_weights)

        gate_history.append(gate_weights.copy())
        dominant_history.append(dom_idx)

        if scene_flags["low_battery"]:
            low_battery_gate_history.append(gate_weights.copy())
            low_battery_dominant_history.append(dom_idx)

        if scene_flags["near_obstacle"]:
            near_obstacle_gate_history.append(gate_weights.copy())
            near_obstacle_dominant_history.append(dom_idx)

        if prev_dom_idx is not None and dom_idx != prev_dom_idx:
            dominant_switches += 1
        prev_dom_idx = dom_idx

        next_obs, reward, terminated, truncated, info = env.step(action)

        ep_return += float(reward)
        step_idx += 1

        battery_ratio = float(info.get("battery_ratio", 1.0))
        on_dock = bool(info.get("on_dock", False))
        coverage = float(info.get("coverage", 0.0))

        if battery_ratio <= cfg.robot.low_battery_threshold:
            low_battery_entered = True

        if low_battery_entered and on_dock:
            returned_on_dock = True

        if step_idx <= max_print_steps:
            gate_str = format_gate_info(model.expert_names, gate_weights)
            print(
                f"[Step {step_idx:04d}] "
                f"act=({action[0]: .3f}, {action[1]: .3f}) "
                f"reward={reward: .3f} "
                f"value={value: .3f} "
                f"battery={battery_ratio:.3f} "
                f"coverage={coverage:.3f} "
                f"ray_min={scene_flags['ray_min']:.3f} "
                f"obs_density={scene_flags['obstacle_density']:.3f} "
                f"low_battery={scene_flags['low_battery']} "
                f"near_obstacle={scene_flags['near_obstacle']} "
                f"on_dock={on_dock} "
                f"dominant={dom_name}({dom_weight:.3f}) "
                f"{gate_str}"
            )

        if render and sleep_time > 0:
            time.sleep(sleep_time)

        obs = next_obs
        done = bool(terminated or truncated)

    env.close()

    num_experts = len(model.expert_names)

    gate_mean = (
        np.mean(np.asarray(gate_history), axis=0)
        if len(gate_history) > 0 else np.zeros(num_experts, dtype=np.float32)
    )

    dominant_counts = np.bincount(
        np.asarray(dominant_history, dtype=np.int64),
        minlength=num_experts
    )

    if len(low_battery_gate_history) > 0:
        low_battery_gate_mean = np.mean(
            np.asarray(low_battery_gate_history), axis=0)
        low_battery_dominant_counts = np.bincount(
            np.asarray(low_battery_dominant_history, dtype=np.int64),
            minlength=num_experts
        )
    else:
        low_battery_gate_mean = np.zeros(num_experts, dtype=np.float32)
        low_battery_dominant_counts = np.zeros(num_experts, dtype=np.int64)

    if len(near_obstacle_gate_history) > 0:
        near_obstacle_gate_mean = np.mean(
            np.asarray(near_obstacle_gate_history), axis=0)
        near_obstacle_dominant_counts = np.bincount(
            np.asarray(near_obstacle_dominant_history, dtype=np.int64),
            minlength=num_experts
        )
    else:
        near_obstacle_gate_mean = np.zeros(num_experts, dtype=np.float32)
        near_obstacle_dominant_counts = np.zeros(num_experts, dtype=np.int64)

    result = {
        "episode_return": float(ep_return),
        "episode_length": int(step_idx),
        "coverage": float(info.get("coverage", 0.0)),
        "done_reason": info.get("done_reason", "unknown"),
        "battery_ratio": float(info.get("battery_ratio", 0.0)),
        "low_battery_entered": bool(low_battery_entered),
        "low_battery_return_success": bool(returned_on_dock),
        "idle_steps": float(info.get("idle_steps", 0.0)),
        "no_progress_steps": float(info.get("no_progress_steps", 0.0)),
        "dock_idle_steps": float(info.get("dock_idle_steps", 0.0)),
        "gate_mean": gate_mean,
        "dominant_counts": dominant_counts,
        "dominant_switches": int(dominant_switches),
        "low_battery_gate_mean": low_battery_gate_mean,
        "low_battery_dominant_counts": low_battery_dominant_counts,
        "near_obstacle_gate_mean": near_obstacle_gate_mean,
        "near_obstacle_dominant_counts": near_obstacle_dominant_counts,
    }
    return result


# =========================================================
# 多回合评测
# =========================================================
@torch.no_grad()
def review_policy(
    checkpoint_path: str,
    episodes: int = 3,
    render: bool = True,
    sleep_time: float = 0.0,
    seed: int = 42,
    force_mode: str = "",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, ckpt = load_model(checkpoint_path, device)

    if force_mode:
        cfg = build_stage_cfg(cfg, force_mode)

    print("=" * 100)
    print("MoE PPO Baseline Review (v3 scene-guided)")
    print("=" * 100)
    print(f"checkpoint      : {checkpoint_path}")
    print(f"device          : {device}")
    print(f"version         : {ckpt.get('version', 'unknown')}")
    print(f"training_mode   : {getattr(cfg, 'training_mode', 'unknown')}")
    print(f"episodes        : {episodes}")
    print(f"render          : {render}")
    print(f"expert_names    : {model.expert_names}")
    print(f"ckpt_update     : {ckpt.get('update', 'unknown')}")
    print(f"ckpt_global_step: {ckpt.get('global_step', 'unknown')}")
    print("=" * 100)

    all_returns = []
    all_coverages = []
    all_return_success = []
    all_idle = []
    all_noprog = []
    all_dock_idle = []
    all_gate_means = []
    all_low_battery_gate_means = []
    all_near_obstacle_gate_means = []
    all_switches = []
    done_reason_count = {}

    dominant_total_counts = np.zeros(len(model.expert_names), dtype=np.int64)
    low_battery_dominant_total_counts = np.zeros(
        len(model.expert_names), dtype=np.int64)
    near_obstacle_dominant_total_counts = np.zeros(
        len(model.expert_names), dtype=np.int64)

    for ep in range(episodes):
        print()
        print(f"{'=' * 38} Episode {ep + 1}/{episodes} {'=' * 38}")

        result = run_one_episode(
            model=model,
            cfg=cfg,
            device=device,
            seed=seed + ep,
            render=render,
            sleep_time=sleep_time,
        )

        all_returns.append(result["episode_return"])
        all_coverages.append(result["coverage"])
        all_return_success.append(
            1.0 if result["low_battery_return_success"] else 0.0)
        all_idle.append(result["idle_steps"])
        all_noprog.append(result["no_progress_steps"])
        all_dock_idle.append(result["dock_idle_steps"])
        all_gate_means.append(result["gate_mean"])
        all_switches.append(result["dominant_switches"])

        dominant_total_counts += result["dominant_counts"]

        if result["low_battery_entered"]:
            all_low_battery_gate_means.append(result["low_battery_gate_mean"])
            low_battery_dominant_total_counts += result["low_battery_dominant_counts"]

        if np.sum(result["near_obstacle_dominant_counts"]) > 0:
            all_near_obstacle_gate_means.append(
                result["near_obstacle_gate_mean"])
            near_obstacle_dominant_total_counts += result["near_obstacle_dominant_counts"]

        dr = result["done_reason"]
        done_reason_count[dr] = done_reason_count.get(dr, 0) + 1

        print("-" * 100)
        print(f"episode_return           : {result['episode_return']:.3f}")
        print(f"episode_length           : {result['episode_length']}")
        print(f"coverage                 : {result['coverage']:.3f}")
        print(f"done_reason              : {result['done_reason']}")
        print(f"final_battery_ratio      : {result['battery_ratio']:.3f}")
        print(f"low_battery_entered      : {result['low_battery_entered']}")
        print(
            f"low_battery_return_succ  : {result['low_battery_return_success']}")
        print(f"idle_steps               : {result['idle_steps']:.1f}")
        print(f"no_progress_steps        : {result['no_progress_steps']:.1f}")
        print(f"dock_idle_steps          : {result['dock_idle_steps']:.1f}")
        print(f"dominant_switches        : {result['dominant_switches']}")

        gate_mean_str = format_gate_info(
            model.expert_names, result["gate_mean"])
        print(f"mean_gate_weights        : {gate_mean_str}")

        dominant_parts = []
        total_dom = max(int(np.sum(result["dominant_counts"])), 1)
        for i, name in enumerate(model.expert_names):
            count_i = int(result["dominant_counts"][i])
            ratio_i = count_i / total_dom
            dominant_parts.append(f"{name}={count_i}({ratio_i:.2%})")
        print(f"dominant_expert_counts   : {' '.join(dominant_parts)}")

        if result["low_battery_entered"]:
            low_gate_str = format_gate_info(
                model.expert_names, result["low_battery_gate_mean"])
            print(f"low_battery_gate_mean    : {low_gate_str}")

            low_dom_parts = []
            total_low_dom = max(
                int(np.sum(result["low_battery_dominant_counts"])), 1)
            for i, name in enumerate(model.expert_names):
                count_i = int(result["low_battery_dominant_counts"][i])
                ratio_i = count_i / total_low_dom
                low_dom_parts.append(f"{name}={count_i}({ratio_i:.2%})")
            print(f"low_battery_dom_counts   : {' '.join(low_dom_parts)}")
        else:
            print("low_battery_gate_mean    : N/A")
            print("low_battery_dom_counts   : N/A")

        if np.sum(result["near_obstacle_dominant_counts"]) > 0:
            near_gate_str = format_gate_info(
                model.expert_names, result["near_obstacle_gate_mean"])
            print(f"near_obstacle_gate_mean  : {near_gate_str}")

            near_dom_parts = []
            total_near_dom = max(
                int(np.sum(result["near_obstacle_dominant_counts"])), 1)
            for i, name in enumerate(model.expert_names):
                count_i = int(result["near_obstacle_dominant_counts"][i])
                ratio_i = count_i / total_near_dom
                near_dom_parts.append(f"{name}={count_i}({ratio_i:.2%})")
            print(f"near_obstacle_dom_counts : {' '.join(near_dom_parts)}")
        else:
            print("near_obstacle_gate_mean  : N/A")
            print("near_obstacle_dom_counts : N/A")

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print(
        f"avg_return               : {np.mean(all_returns):.3f} ± {np.std(all_returns):.3f}")
    print(f"avg_coverage             : {np.mean(all_coverages):.3f}")
    print(f"return_success_rate      : {np.mean(all_return_success):.3f}")
    print(f"avg_idle_steps           : {np.mean(all_idle):.2f}")
    print(f"avg_no_progress_steps    : {np.mean(all_noprog):.2f}")
    print(f"avg_dock_idle_steps      : {np.mean(all_dock_idle):.2f}")
    print(f"avg_dominant_switches    : {np.mean(all_switches):.2f}")

    if len(all_gate_means) > 0:
        overall_gate_mean = np.mean(np.asarray(all_gate_means), axis=0)
        print("overall_mean_gate_weights:", format_gate_info(
            model.expert_names, overall_gate_mean))

    total_dom_all = max(int(np.sum(dominant_total_counts)), 1)
    dom_parts = []
    for i, name in enumerate(model.expert_names):
        c = int(dominant_total_counts[i])
        r = c / total_dom_all
        dom_parts.append(f"{name}={c}({r:.2%})")
    print("overall_dominant_counts  :", " ".join(dom_parts))

    if len(all_low_battery_gate_means) > 0:
        overall_low_gate_mean = np.mean(
            np.asarray(all_low_battery_gate_means), axis=0)
        print("low_battery_gate_mean    :", format_gate_info(
            model.expert_names, overall_low_gate_mean))

        total_low_dom_all = max(
            int(np.sum(low_battery_dominant_total_counts)), 1)
        low_dom_parts = []
        for i, name in enumerate(model.expert_names):
            c = int(low_battery_dominant_total_counts[i])
            r = c / total_low_dom_all
            low_dom_parts.append(f"{name}={c}({r:.2%})")
        print("low_battery_dom_counts   :", " ".join(low_dom_parts))
    else:
        print("low_battery_gate_mean    : N/A")
        print("low_battery_dom_counts   : N/A")

    if len(all_near_obstacle_gate_means) > 0:
        overall_near_gate_mean = np.mean(
            np.asarray(all_near_obstacle_gate_means), axis=0)
        print("near_obstacle_gate_mean  :", format_gate_info(
            model.expert_names, overall_near_gate_mean))

        total_near_dom_all = max(
            int(np.sum(near_obstacle_dominant_total_counts)), 1)
        near_dom_parts = []
        for i, name in enumerate(model.expert_names):
            c = int(near_obstacle_dominant_total_counts[i])
            r = c / total_near_dom_all
            near_dom_parts.append(f"{name}={c}({r:.2%})")
        print("near_obstacle_dom_counts :", " ".join(near_dom_parts))
    else:
        print("near_obstacle_gate_mean  : N/A")
        print("near_obstacle_dom_counts : N/A")

    print("done_reason_count        :", done_reason_count)
    print("=" * 100)


# =========================================================
# 命令行参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints_moeppo/latest_model.pt",
        help="模型路径，可传 pt 文件或目录"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="回测 episode 数"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="是否开启可视化"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="每步可视化暂停秒数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="起始随机种子"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="",
        choices=["", "baseline_easy", "baseline_mid", "baseline_full"],
        help="可选：强制测试环境模式"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = resolve_checkpoint_path(args.ckpt)

    review_policy(
        checkpoint_path=ckpt_path,
        episodes=args.episodes,
        render=args.render,
        sleep_time=args.sleep,
        seed=args.seed,
        force_mode=args.mode,
    )
