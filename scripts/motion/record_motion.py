# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
# make scripts/rsl_rl importable when this script is executed from scripts/motion
_RSL_RL_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "rsl_rl"
if str(_RSL_RL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_SCRIPTS_DIR))

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

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
from tqdm import tqdm

from rsl_rl.runners import *
from rsl_rl.utils.exporter import attach_onnx_metadata, export_policy_as_onnx

mujoco2isaaclab_joint = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
isaaclab2mujoco_joint = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]

mujoco2isaaclab_keypose = [0, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 6, 12, 18, 23]
isaaclab2mujoco_keypose = [0, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 3, 6, 10, 14, 18, 22, 7, 11, 15, 19, 23]


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
    env_cfg.episode_length_s = 100.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.viewer.origin_type = None
    env_cfg.viewer.asset_name = None

    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    env_cfg.events.randomize_joint_parameters = None
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.events.push_robot = None

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    # 关闭依赖 terrain_generator 的地形课程项，避免 reset 时访问 None
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None

    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (0.1, 0.1)

    # env_cfg.commands.base_velocity.resampling_time_range = (5.0, 5.0)
    # env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.zero_prob = [0.0, 0.0, 0.2]

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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.class_name)
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

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(policy_nn, path=export_model_dir, filename="policy.onnx")
    attach_onnx_metadata(env=env.unwrapped, path=export_model_dir)

    dt = env.unwrapped.step_dt
    record_duration = 20  # s
    stop_state_log = round(record_duration / dt)

    # reset environment
    obs = env.get_observations()
    target_joint_names = [
        "l_leg_pitch_joint",
        "l_leg_roll_joint",
        "l_leg_yaw_joint",
        "l_knee_pitch_joint",
        "l_ankle_pitch_joint",
        "l_ankle_roll_joint",
        "r_leg_pitch_joint",
        "r_leg_roll_joint",
        "r_leg_yaw_joint",
        "r_knee_pitch_joint",
        "r_ankle_pitch_joint",
        "r_ankle_roll_joint",
        "waist_yaw_joint",
        "l_arm_pitch_joint",
        "l_arm_roll_joint",
        "l_arm_yaw_joint",
        "l_elbow_pitch_joint",
        "r_arm_pitch_joint",
        "r_arm_roll_joint",
        "r_arm_yaw_joint",
        "r_elbow_pitch_joint",
    ]
    target_keypos_names = [
        "base_link",
        "l_leg_yaw_link",
        "l_leg_roll_link",
        "l_leg_pitch_link",
        "l_knee_pitch_link",
        "l_ankle_pitch_link",
        "l_ankle_roll_link",
        "r_leg_yaw_link",
        "r_leg_roll_link",
        "r_leg_pitch_link",
        "r_knee_pitch_link",
        "r_ankle_pitch_link",
        "r_ankle_roll_link",
        "waist_yaw_link",
        "l_arm_pitch_link",
        "l_arm_roll_link",
        "l_arm_yaw_link",
        "l_elbow_pitch_link",
        "r_arm_pitch_link",
        "r_arm_roll_link",
        "r_arm_yaw_link",
        "r_elbow_pitch_link",
    ]

    robot_joint_names = list(env.unwrapped.robot.joint_names)
    robot_body_names = list(env.unwrapped.robot.body_names)
    joint_reorder_idx = [robot_joint_names.index(name) for name in target_joint_names]
    keypos_reorder_idx = [robot_body_names.index(name) for name in target_keypos_names]

    motion_frames = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        for i in tqdm(range(stop_state_log)):
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)


                # if i < 200:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 0] = 0.0
                # elif i < 500:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = 1.0
                # elif i < 1500:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = -1.0
                # # elif i < 1250:
                # #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = 0.0
                # elif i < 2500:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = -1.0
                # elif i < 3000:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = 0.0
                # # elif i < 2500:
                # #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 0] = 6.0
                # # elif i < 3000:
                # #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 0] = 1.5
                # else:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:,2] = 0.0
    
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = 0.0
                env.unwrapped.command_manager._terms["base_velocity"].command[:, 0] = 0.8
                if 800 >= i > 600:
                    env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = 1.0
                elif i > 800:
                    env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = -1.0
                # if 2000 < i < 2500:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = 1.5
                # elif 2500 < i < 3000:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = -1.5
                # else:
                # env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = 0.0

                env.unwrapped.command_manager._terms["base_velocity"].command[:, 1] = 0.0
                
                env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = 0.0
                # env stepping
                obs, _, _, _ = env.step(actions)

                root_pos = env.unwrapped.robot.data.root_pos_w.clone()
                root_quat = env.unwrapped.robot.data.root_quat_w.clone()
                root_quat = torch.cat((root_quat[:, 1:], root_quat[:, 0].unsqueeze(1)), dim=1)  #  wxyz -> xyzw
                dof_pos = env.unwrapped.robot.data.joint_pos[:, joint_reorder_idx]
                key_pos = (env.unwrapped.robot.data.body_pos_w[:, keypos_reorder_idx, :] - root_pos.unsqueeze(1)).flatten(1, 2)

                # key_pos = (env.unwrapped.robot.data.body_pos_w).flatten(1, 2)
                root_vel = env.unwrapped.robot.data.root_lin_vel_b.clone()
                root_ang_vel = env.unwrapped.robot.data.root_ang_vel_b.clone()
                dof_vel = env.unwrapped.robot.data.joint_vel[:, joint_reorder_idx]
                contact_mask = obs["critic"][:, 74:76].clone()
                cur_frames_tensor = torch.cat(
                    (root_pos, root_quat, dof_pos, key_pos, root_vel, root_ang_vel, dof_vel, contact_mask), dim=-1
                )
                cur_frames = cur_frames_tensor[0].tolist()
                motion_frames.append(cur_frames)

        motion_data = {}
        motion_data["FrameDuration"] = dt
        motion_data["JointNames"] = target_joint_names
        motion_data["KeyPosNames"] = target_keypos_names
        motion_data["MotionWeight"] = 0.5
        motion_data["Frames"] = motion_frames[200:]
        # export motion capture to json format
        with open("../../source/NoetixBumi/NoetixBumi/assets/datasets/amp_data/bumi/bumi_processed/walk_v2.json", "w") as f:
            json.dump(motion_data, f, indent=1, separators=(",", ":"))
            print("Generating " + "walk_v2.json")
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
