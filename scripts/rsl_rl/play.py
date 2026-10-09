# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import threading

try:
    from pynput import keyboard
except Exception:
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
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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

import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import NoetixBumi.tasks  # noqa: F401
import NoetixBumi.terrains as terrain_gen  # 用于指定地形类型（取消注释下方代码时使用）
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rsl_rl.runners as rsl_rl_runners
from rsl_rl.utils.exporter import attach_onnx_metadata, export_policy_as_onnx

from NoetixBumi.tasks.walk_rand_arms.noetix_bumi.mdp.commands import KeyboardVelocityCommand


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set the play environment config
    env_cfg.episode_length_s = 200.0
    env_cfg.scene.num_envs = 10
    env_cfg.scene.env_spacing = 2.5

    # WalkRun/AMP has a base_velocity command; Mimic uses commands.motion instead.
    base_velocity_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if base_velocity_cfg is not None:
        base_velocity_cfg.ranges.lin_vel_x = (0.0, 0.0)
        base_velocity_cfg.ranges.lin_vel_y = (0.0, 0.0)
        base_velocity_cfg.ranges.ang_vel_z = (1.2, 1.2)
        base_velocity_cfg.ranges.zero_prob = [0.0, 0.0, 0.0]
    env_cfg.observations.policy.enable_corruption = False

    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    env_cfg.events.randomize_joint_parameters = None
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.events.push_robot = None

    # env_cfg.scene.terrain.terrain_type = "generator"
    # env_cfg.scene.terrain.terrain_generator = None

    # if env_cfg.scene.terrain.terrain_generator is not None:
    #     env_cfg.scene.terrain.terrain_generator = terrain_gen.GRAVEL_TERRAINS_CFG_V4

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Keep the camera attached to the robot when recording video; otherwise the frame can be empty in headless mode.
    if args_cli.video:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
    else:
        env_cfg.viewer.origin_type = None
        env_cfg.viewer.asset_name = None

    if CONTROL_ROBOT:
        if base_velocity_cfg is None:
            print("[WARN] Keyboard control requested but base_velocity command not found in env config.")
        else:
            base_velocity_cfg.class_type = KeyboardVelocityCommand
            base_velocity_cfg.resampling_time_range = (1e9, 1e9)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    command_listener = None
    command_lock = threading.Lock()
    reset_env_flag = threading.Event()
    command_term = None
    if CONTROL_ROBOT:
        try:
            command_term = env.unwrapped.command_manager.get_term("base_velocity")
        except KeyError:
            command_term = None
            print("[WARN] base_velocity term not found in environment; keyboard control disabled.")
        if command_term is not None and not isinstance(command_term, KeyboardVelocityCommand):
            print("[WARN] base_velocity term is not KeyboardVelocityCommand; keyboard control disabled.")
            command_term = None

        if command_term is not None:
            if keyboard is None:
                raise ImportError("Keyboard control requires 'pynput'. Install via 'pip install pynput'.")
            step_size = 0.1
            command_term.reset_manual_command()

            print("[INFO] Keyboard control enabled (W/S forward/back, A/D left/right, Q/E yaw, R reset command, T reset robot).")

            def _print_command():
                cmd = command_term.get_manual_command()[0].detach().cpu().tolist()
                print(f"[CMD] Current command (x, y, yaw): {cmd}")

            def _adjust(idx: int, delta: float):
                with command_lock:
                    command_term.adjust_manual_command_component(idx, delta)
                _print_command()

            def _reset():
                with command_lock:
                    command_term.reset_manual_command()
                _print_command()

            def _on_press(key_event):
                try:
                    char = key_event.char.lower()
                except AttributeError:
                    return
                if char == "8":
                    _adjust(0, step_size)
                elif char == "2":
                    _adjust(0, -step_size)
                elif char == "6":
                    _adjust(1, -step_size)
                elif char == "4":
                    _adjust(1, step_size)
                elif char == "7":
                    _adjust(2, step_size)
                elif char == "9":
                    _adjust(2, -step_size)
                elif char == "r":
                    _reset()
                elif char == "t":
                    reset_env_flag.set()
                    print("[INFO] Robot reset requested (T key pressed).")

            command_listener = keyboard.Listener(on_press=_on_press)
            command_listener.daemon = True
            command_listener.start()

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    try:
        runner_class = getattr(rsl_rl_runners, agent_cfg.class_name)
    except AttributeError as exc:
        raise ValueError(f"Runner class '{agent_cfg.class_name}' not found in rsl_rl.runners.") from exc
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        attach_onnx_metadata(env=env.unwrapped, path=export_model_dir)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # Check for environment reset request (before inference_mode)
            if reset_env_flag.is_set():
                reset_env_flag.clear()
                print("[INFO] Resetting environment...")
                obs, _ = env.reset()
                if command_term is not None:
                    command_term.reset_manual_command()
                    print("[INFO] Command reset to zero.")
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            # print(f"Command: {env.unwrapped.command_manager.get_command('base_velocity').cpu().numpy()[0]}, Current Velocity: {env.unwrapped.robot.data.root_lin_vel_b.cpu().numpy()[0]}")
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            print(f"[INFO] Recording video: {timestep}/{args_cli.video_length}")
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if command_listener is not None:
        command_listener.stop()


if __name__ == "__main__":
    EXPORT_POLICY = True
    CONTROL_ROBOT = keyboard is not None and not args_cli.headless
    # run the main function
    main()
    # close sim app
    simulation_app.close()
