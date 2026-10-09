# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import select
import sys
import threading

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--print_joint_order",
    action="store_true",
    help="Print the Isaac Sim articulation joint order after the environment is created.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keyboard_control",
    action="store_true",
    default=True,
    help="Enable keyboard controls: SPACE pause/resume, N next motion, B previous motion, R restart current motion.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import NoetixBumi.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl.runners import *


@hydra_task_config(args_cli.task, args_cli.agent)
def play_animation(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.viewer.origin_type = None
    env_cfg.viewer.asset_name = None
    env_cfg.commands.base_velocity.debug_vis = True

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.print_joint_order:
        robot = env.unwrapped.robot
        print("\n[INFO] Isaac Sim articulation joint order:")
        for joint_index, joint_name in enumerate(robot.joint_names):
            print(f"  [{joint_index:2d}] {joint_name}")
        print(f"[INFO] joint_pos shape: {tuple(robot.data.joint_pos.shape)}")
        print(f"[INFO] joint_vel shape: {tuple(robot.data.joint_vel.shape)}\n")

    # ------ visualize -----
    # 定义两种可视化标记：
    # - "motion": 索引 0, 红色球体，用于显示参考动作的关键点位置
    # - "robot": 索引 1, 蓝色球体，用于显示机器人刚体的实际位置
    mimic_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TrackingMarker",
        markers={
            "motion": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # 红色
            ),
            "robot": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),  # 蓝色
            ),
        },
    )
    rigid_body_pos = env.unwrapped.robot.data.body_pos_w
    tracking_markers = VisualizationMarkers(mimic_marker_cfg)

    # 设置标记索引：决定每个位置使用哪种标记类型
    # translations 的拼接顺序是: [rigid_body_pos, key_pos]
    # 所以：
    # - 前半部分（rigid_body_pos）应该使用 "robot"（索引 1，蓝色）
    # - 后半部分（key_pos）应该使用 "motion"（索引 0，红色）
    num_bodies = rigid_body_pos.shape[1]
    marker_indices = torch.zeros(num_bodies * 2, dtype=torch.long, device=env.unwrapped.device)
    marker_indices[:num_bodies] = 1  # 前半部分：机器人位置 → "robot"（蓝色）
    marker_indices[num_bodies:] = 0  # 后半部分：参考动作位置 → "motion"（红色）

    frame_cnt = 0
    motion_index = 0  # 当前播放的motion索引

    playback_lock = threading.Lock()
    is_paused = False
    should_exit = False
    control_flags = {
        "toggle_pause": False,
        "next": False,
        "prev": False,
        "restart": False,
        "quit": False,
    }

    def _set_flag(name: str):
        with playback_lock:
            control_flags[name] = True

    def _apply_pending_controls():
        nonlocal is_paused, motion_index, frame_cnt, should_exit
        with playback_lock:
            if control_flags["quit"]:
                should_exit = True
                control_flags["quit"] = False

            if control_flags["toggle_pause"]:
                is_paused = not is_paused
                state = "paused" if is_paused else "playing"
                print(f"[INFO] Playback {state}.")
                control_flags["toggle_pause"] = False

            if control_flags["next"]:
                motion_index = (motion_index + 1) % len(env.unwrapped.motion_loader.trajectory_num_frames)
                frame_cnt = 0
                print(f"[INFO] Switched to next motion: {motion_index}.")
                control_flags["next"] = False

            if control_flags["prev"]:
                motion_index = (motion_index - 1) % len(env.unwrapped.motion_loader.trajectory_num_frames)
                frame_cnt = 0
                print(f"[INFO] Switched to previous motion: {motion_index}.")
                control_flags["prev"] = False

            if control_flags["restart"]:
                frame_cnt = 0
                print(f"[INFO] Restarted current motion: {motion_index}.")
                control_flags["restart"] = False

    def _poll_terminal_control():
        # Non-blocking stdin polling works better than input() in background threads.
        if not sys.stdin or not sys.stdin.isatty():
            return
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return
        cmd = sys.stdin.readline().strip().lower()
        if cmd == "p":
            _set_flag("toggle_pause")
        elif cmd == "n":
            _set_flag("next")
        elif cmd == "b":
            _set_flag("prev")
        elif cmd == "r":
            _set_flag("restart")
        # elif cmd == "q":
        #     _set_flag("quit")

    listener = None
    if args_cli.keyboard_control:
        if keyboard is None:
            print("[WARN] pynput is unavailable. Falling back to terminal controls only.")
        else:
            print("[INFO] Keyboard control enabled: SPACE pause/resume, N next, B previous, R restart, Q quit.")

            def _on_press(key):
                try:
                    k = key.char.lower() if hasattr(key, "char") and key.char is not None else None
                except Exception:
                    k = None

                if key == keyboard.Key.space:
                    _set_flag("toggle_pause")
                elif k == "n":
                    _set_flag("next")
                elif k == "b":
                    _set_flag("prev")
                elif k == "r":
                    _set_flag("restart")
                # elif k == "q":
                #     _set_flag("quit")

            try:
                listener = keyboard.Listener(on_press=_on_press)
                listener.start()
            except Exception as err:
                listener = None
                print(f"[WARN] Failed to start pynput keyboard listener: {err}")
                print("[WARN] Use terminal controls instead (p/n/b/r/q).")

        print("[INFO] Terminal control enabled: input p/n/b/r/q then press Enter.")

    while simulation_app.is_running():
        _poll_terminal_control()
        _apply_pending_controls()
        if should_exit:
            break

        with playback_lock:
            current_motion_frames = env.unwrapped.motion_loader.trajectory_num_frames[motion_index]
            frame_duration = env.unwrapped.motion_loader.trajectory_frame_durations[motion_index]
            local_frame_cnt = frame_cnt
            local_motion_index = motion_index
            local_is_paused = is_paused

        time = (local_frame_cnt % (current_motion_frames - 1)) * frame_duration
        key_pos = env.unwrapped.visualize_motion(time, local_motion_index)

        with playback_lock:
            if not local_is_paused:
                frame_cnt += 1

            if frame_cnt >= current_motion_frames:
                frame_cnt = 0  # 重置帧计数器
                motion_index = (motion_index + 1) % len(env.unwrapped.motion_loader.trajectory_num_frames)

        # 获取当前位置数据
        rigid_body_pos = env.unwrapped.robot.data.body_pos_w.clone().view(-1, 3)  # 机器人刚体位置
        key_pos = key_pos.view(-1, 3)  # 参考动作关键点位置（从 visualize_motion 返回）

        # 合并位置数据：前半部分是机器人位置（蓝色），后半部分是参考动作位置（红色）
        translations = torch.cat((rigid_body_pos, key_pos), dim=0)

        # 可视化标记：红色球体显示参考动作，蓝色球体显示机器人实际位置
        tracking_markers.visualize(translations=translations, marker_indices=marker_indices)

    if listener is not None:
        listener.stop()


if __name__ == "__main__":
    # run the main function
    play_animation()
    # close sim app
    simulation_app.close()
