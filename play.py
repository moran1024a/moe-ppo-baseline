import argparse
import time
from typing import Tuple

import numpy as np
import pygame

from env_cfg import EnvConfig
from cleaning_robot_env import CleaningRobotEnv


def get_keyboard_action() -> Tuple[np.ndarray, bool, bool]:
    """
    返回:
    - action: np.ndarray([v, w], dtype=float32)
    - reset_pressed: 是否按下 R
    - quit_pressed: 是否退出
    """
    reset_pressed = False
    quit_pressed = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_pressed = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                reset_pressed = True
            elif event.key == pygame.K_ESCAPE:
                quit_pressed = True

    keys = pygame.key.get_pressed()

    # 线速度控制：W 前进，S 后退
    v = 0.0
    if keys[pygame.K_w]:
        v += 1.0
    if keys[pygame.K_s]:
        v -= 1.0

    # 角速度控制：A 左转，D 右转
    # 注意：如果你觉得左右反了，把下面两个符号交换即可
    w = 0.0
    if keys[pygame.K_a]:
        w -= 1.0
    if keys[pygame.K_d]:
        w += 1.0

    # 空格急停
    if keys[pygame.K_SPACE]:
        v = 0.0
        w = 0.0

    action = np.array([v, w], dtype=np.float32)
    return action, reset_pressed, quit_pressed


def print_help() -> None:
    print("=" * 72)
    print("键盘控制说明")
    print("=" * 72)
    print("W / S : 前进 / 后退")
    print("A / D : 左转 / 右转")
    print("SPACE : 停止")
    print("R     : 重开当前回合")
    print("ESC   : 退出")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=999999)
    parser.add_argument("--frame-skip", type=int, default=1, help="每次按键动作重复执行多少步")
    args = parser.parse_args()

    cfg = EnvConfig()
    env = CleaningRobotEnv(cfg=cfg, render_mode="human")

    print_help()

    obs, info = env.reset(seed=args.seed)
    print("[RESET]", info)

    episode = 1
    running = True

    last_info_print_time = 0.0
    info_print_interval = 0.2  # 秒

    while running and episode <= args.max_episodes:
        action, reset_pressed, quit_pressed = get_keyboard_action()

        if quit_pressed:
            break

        if reset_pressed:
            obs, info = env.reset()
            print(f"\n[RESET] episode={episode}")
            print(info)
            continue

        terminated = False
        truncated = False
        reward = 0.0

        for _ in range(max(1, args.frame_skip)):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        now = time.time()
        if now - last_info_print_time >= info_print_interval:
            print(
                f"\rstep={info.get('steps', 0):4d} | "
                f"reward={reward:7.3f} | "
                f"coverage={info.get('coverage', 0.0):.3f} | "
                f"battery={info.get('battery', 0.0):6.2f} | "
                f"on_dock={int(info.get('on_dock', False))} | "
                f"done={info.get('done_reason', None)}      ",
                end="",
                flush=True,
            )
            last_info_print_time = now

        if terminated or truncated:
            print("\n" + "-" * 72)
            print(f"[EPISODE END] episode={episode}")
            print(f"terminated={terminated}, truncated={truncated}")
            print(info)
            print("-" * 72)

            episode += 1
            obs, info = env.reset()
            print(f"[AUTO RESET] episode={episode}")
            print(info)

    env.close()
    print("\n已退出。")


if __name__ == "__main__":
    main()