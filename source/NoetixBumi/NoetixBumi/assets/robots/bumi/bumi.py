# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import ArticulationCfg
from NoetixBumi.actuators import DelayedImplicitActuatorCfg
from NoetixBumi.assets import ASSET_DIR

ARMATURE_431025 =  0.0018606249999999999
ARMATURE_431536 =  0.007642512
ARMATURE_431040 =  0.0045024
ARMATURE_DM4340 = 0.032
ARMATURE_YKS4315 = 0.033048
ARMATURE_LZ00 =  0.0007
ARMATURE_LZ05 =  0.001

inertia_arm = ARMATURE_LZ05 # arm
inertia_waist_yaw = ARMATURE_431040 # waist 
inertia_hip_yaw = 0.000107 +  ARMATURE_431025 # hip yaw  拟合优度 (R²) : 0.9935  巨小
inertia_hip_roll = 0.209604  + ARMATURE_431536 # hip roll    拟合优度 (R²) : 0.60
inertia_hip_pitch = 0.319813  + ARMATURE_431536  # hip pitch    拟合优度 (R²) : 0.9884
inertia_knee = 0.028678  + ARMATURE_431536 # knee    拟合优度 (R²) : 0.8447
inertia_ankle_pitch_real = 0.021115 # ankle_pitch    拟合优度 (R²) : 0.8645
inertia_ankle_roll_real = 0.002829 # ankle_roll    拟合优度 (R²) : 0.6867

NATURAL_FREQ_1 = 3 * 2.0 * 3.1415926535  #  for 3Hz
NATURAL_FREQ_2 = 4 * 2.0 * 3.1415926535  #  for 4Hz
NATURAL_FREQ_3 = 5 * 2.0 * 3.1415926535  #  for 5Hz

NATURAL_FREQ_4 = 10 * 2.0 * 3.1415926535  #  for 5Hz

STIFFNESS_arm = inertia_arm * NATURAL_FREQ_4**2 * 3  
STIFFNESS_waist = inertia_waist_yaw * NATURAL_FREQ_4**2 * 3
STIFFNESS_hip_yaw = inertia_hip_yaw * NATURAL_FREQ_4**2 * 2

STIFFNESS_hip_roll = inertia_hip_roll * NATURAL_FREQ_1**2
STIFFNESS_hip_pitch = inertia_hip_pitch * NATURAL_FREQ_1**2

STIFFNESS_knee = inertia_knee * NATURAL_FREQ_3**2 * 2

STIFFNESS_ankle_pitch = inertia_ankle_pitch_real * NATURAL_FREQ_1**2
STIFFNESS_ankle_roll = inertia_ankle_roll_real * NATURAL_FREQ_1**2



DAMPING_arm = 2 * inertia_arm * NATURAL_FREQ_4 * 3
DAMPING_waist = 2 * inertia_waist_yaw * NATURAL_FREQ_4 * 6
DAMPING_hip_yaw = 2 * inertia_hip_yaw * NATURAL_FREQ_4 * 8

DAMPING_hip_roll = 0.7 * inertia_hip_roll * NATURAL_FREQ_1
DAMPING_hip_pitch = 0.7 * inertia_hip_pitch * NATURAL_FREQ_1

DAMPING_knee = 0.9 * inertia_knee * NATURAL_FREQ_3 * 2

DAMPING_ankle_pitch = 0.9 * inertia_ankle_pitch_real * NATURAL_FREQ_1
DAMPING_ankle_roll = 0.9 * inertia_ankle_roll_real * NATURAL_FREQ_1



BUMI_21DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/robots/bumi/urdf/bumi.urdf",
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            "l_leg_yaw_joint": 0.0,
            "l_leg_roll_joint": 0.0,
            "l_leg_pitch_joint": -0.1495,
            "l_knee_pitch_joint": 0.3215,
            "l_ankle_pitch_joint": -0.1720,
            "l_ankle_roll_joint": 0.0,
            "r_leg_yaw_joint": 0.0,
            "r_leg_roll_joint": 0.0,
            "r_leg_pitch_joint": -0.1495,
            "r_knee_pitch_joint": 0.3215,
            "r_ankle_pitch_joint": -0.1720,
            "r_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "l_arm_pitch_joint": 0.0,
            "l_arm_roll_joint": 0.3,
            "l_arm_yaw_joint": 0.0,
            "l_elbow_pitch_joint": 0.0,
            "r_arm_pitch_joint": 0.0,
            "r_arm_roll_joint": -0.3,
            "r_arm_yaw_joint": 0.0,
            "r_elbow_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_yaw_joint",
                ".*_leg_roll_joint",
                ".*_leg_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_leg_yaw_joint": 27.0,
                ".*_leg_roll_joint": 27.0,
                ".*_leg_pitch_joint": 60.0,
                ".*_knee_pitch_joint": 60.0,
            },
            velocity_limit_sim={
                ".*_leg_yaw_joint": 9.0,
                ".*_leg_roll_joint": 12.0,
                ".*_leg_pitch_joint": 12.0,
                ".*_knee_pitch_joint": 12.0,
            },
            stiffness={
                ".*_leg_yaw_joint": 60,
                ".*_leg_roll_joint": 60,
                ".*_leg_pitch_joint": 60,
                ".*_knee_pitch_joint": 45,
            },
            damping={
                ".*_leg_yaw_joint": 2.5,
                ".*_leg_roll_joint": 3.0,
                ".*_leg_pitch_joint": 3.0,
                ".*_knee_pitch_joint": 2.0,
            },
            min_delay=0,
            max_delay=4,
        ),
        "waist": DelayedImplicitActuatorCfg(
            effort_limit_sim=27.0,
            velocity_limit_sim=9.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=53,
            damping=3.4,
            min_delay=0,
            max_delay=4,
        ),
        "feet": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=13.0,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_ankle_pitch_joint": 10,
                ".*_ankle_roll_joint": 10,
            },
            damping={
                ".*_ankle_pitch_joint": 0.5,
                ".*_ankle_roll_joint": 0.5,
            },
            # friction={
            #     ".*_ankle_pitch_joint": 1.930293,
            #     ".*_ankle_roll_joint": 0.735826,
            # },
            armature={
                ".*_ankle_pitch_joint": 0.01,
                ".*_ankle_roll_joint": 0.01,
            },
            # viscous_friction={
            #     ".*_ankle_pitch_joint": 0.099857,
            #     ".*_ankle_roll_joint": 0.269741,
            # },
            min_delay=0,
            max_delay=4,
        ),
        "arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit_sim=5.5,
            velocity_limit_sim=50,
            stiffness=12,
            damping=0.4,
            # armature=ARMATURE_LZ05,
            min_delay=0,
            max_delay=4,
        ),
    },
)
BUMI_21DOF_NAMES=[
     'l_leg_pitch_joint',
     'r_leg_pitch_joint',
     'waist_yaw_joint',
     'l_leg_roll_joint',
     'r_leg_roll_joint',
     'l_arm_pitch_joint',
     'r_arm_pitch_joint',
     'l_leg_yaw_joint',
     'r_leg_yaw_joint',
     'l_arm_roll_joint',
     'r_arm_roll_joint',
     'l_knee_pitch_joint',
     'r_knee_pitch_joint',
     'l_arm_yaw_joint',
     'r_arm_yaw_joint',
     'l_ankle_pitch_joint',
     'r_ankle_pitch_joint',
     'l_elbow_pitch_joint',
     'r_elbow_pitch_joint',
     'l_ankle_roll_joint',
     'r_ankle_roll_joint'
     ]

BUMI_21DOF_ACTION_SCALE = {}
for a in BUMI_21DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            BUMI_21DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


