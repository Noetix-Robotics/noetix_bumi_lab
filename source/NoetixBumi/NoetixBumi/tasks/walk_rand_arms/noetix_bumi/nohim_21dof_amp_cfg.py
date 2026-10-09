# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""21DOF BUMI walk-run AMP task with randomized arm dynamics."""

import glob

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from NoetixBumi.assets import ASSET_DIR
from NoetixBumi.assets.robots.bumi.bumi import BUMI_21DOF_ACTION_SCALE, BUMI_21DOF_CFG, BUMI_21DOF_NAMES
from . import mdp


FULL_BODY_JOINTS = BUMI_21DOF_NAMES
FULL_BODY_BODIES = [
    "base_link",
    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
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


@configclass
class NoHim21DofSceneCfg(InteractiveSceneCfg):
    """仿真场景、机器人、光照和接触传感器。"""

    terrain = TerrainImporterCfg(
        # 地面在 USD Stage 中的路径。
        prim_path="/World/ground",
        # plane 表示无限平面，不使用程序化复杂地形。
        terrain_type="plane",
        # 平地不需要 TerrainGeneratorCfg。
        terrain_generator=None,
        # -1 允许该地面与所有碰撞组发生碰撞。
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # multiply：地面与机器人材料系数相乘得到最终接触参数。
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # 静摩擦决定站立和蹬地时开始滑动的难易程度。
            static_friction=0.8,
            # 动摩擦决定脚已经滑动后的摩擦阻力。
            dynamic_friction=0.7,
        ),
        visual_material=sim_utils.MdlFileCfg(
            # 仅影响显示，不影响接触动力学。
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,  # 自动生成地面纹理坐标。
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,  # 不显示地形调试标记。
    )
    # 物理上仍加载完整 21DOF BUMI3；Actor 只控制其中 21DOF。
    robot: ArticulationCfg = BUMI_21DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 主方向光，只影响画面亮度。
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # 环境天空光和反射贴图，只影响渲染。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # 为机器人所有刚体创建接触传感器，奖励项再按 body_names 选择脚或其他部位。
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        # 保存最近 3 个物理步的接触数据，供冲击和接触历史函数使用。
        history_length=3,
        # 记录离地持续时间，feet_air_time 奖励依赖该数据。
        track_air_time=True,
        # 接触力超过 10N 时传感器认为发生有效接触。
        force_threshold=10.0,
        debug_vis=False,
    )
    # 纯平地任务不使用高度扫描器，因此 policy 没有地形高度观测。
    height_scanner = None


@configclass
class NoHim21DofCommandsCfg:
    """速度命令采样范围。命令格式为 [vx, vy, wz]。"""

    base_velocity = mdp.UniformVelocityWithZeroCommandCfg(
        asset_name="robot",
        # 每个环境保持一条命令 6~15 秒，然后重新采样。
        resampling_time_range=(6.0, 15.0),
        # 20% 环境直接采样完整零命令，用于训练稳定站立。
        rel_standing_envs=0.10,
        debug_vis=False,
        ranges=mdp.UniformVelocityWithZeroCommandCfg.Ranges(
            # 机体朝向坐标系 x：负值后退，正值前进，单位 m/s。
            lin_vel_x=(-1.0, 1.2),
            # 机体朝向坐标系 y：负/正分别表示两个侧移方向，单位 m/s。
            lin_vel_y=(-0.7, 0.7),
            # 世界竖直轴偏航角速度命令，单位 rad/s。
            ang_vel_z=(-1.0, 1.0),
            # vx、vy、wz 各有20%概率独立清零，同时覆盖混合命令和纯方向命令。
            # 另有20%环境由 rel_standing_envs 直接设为完整零命令。
            zero_prob=[0.2, 0.2, 0.2],
        ),
    )


@configclass
class NoHim21DofBaseEventCfg:
    """Domain randomization、AMP 初始状态重置和外部扰动。"""

    # startup：环境创建时执行一次，为不同并行环境分配不同接触材料。
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            # 对机器人所有刚体碰撞材料生效。
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 静摩擦、动摩擦和恢复系数的均匀采样范围。
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            # 先建立64组离散材料，环境从桶中采样，降低 GPU 材料数量开销。
            "num_buckets": 64,
        },
    )
    # startup：给21DOF默认角度增加制造误差/零位误差，每个环境固定一组。
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS),
            "pos_distribution_params": (-0.1, 0.1),  # 默认角附加 ±0.1rad 偏移。
            "operation": "add",  # add 表示加法，不是乘法缩放。
        },
    )
    # startup：随机移动腰部刚体质心，模拟装配误差和载荷位置变化。
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            # 在刚体局部坐标系 xyz 三轴分别加入 ±5cm 质心偏移。
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    # startup：在腰部质量上增加随机载荷。
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_link"]),
            "mass_distribution_params": (-2.5, 3.0),  # 相对标称值增加 -2.5~3.0kg。
            "operation": "add",
        },
    )
    # startup：将全部关节 effort limit 乘以随机比例，模拟电机输出能力差异。
    randomize_actuator_effort_limit = EventTerm(
        func=mdp.randomize_actuator_effort_limit,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "effort_limit_distribution_params": (0.8, 1.2),  # 标称限制的80%~120%。
            "operation": "scale",  # scale 表示乘法缩放。
        },
    )
    # reset：每个 episode 重置根节点和21DOF关节状态。
    reset_robot_states = EventTerm(
        func=mdp.reset_root_state_amp,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS),
            # True：允许直接从 AMP motion 中抽取参考帧作为初始状态。
            "reference_state_initialization": True,
            # 80% reset 使用参考状态初始化（RSI），其余使用默认状态附近随机初始化。
            "prob_rsi": 0.8,
            # 根位置扰动：xy 平移 ±0.5m，yaw 覆盖近似完整一圈。
            "root_pos_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # 在所选初始状态的根线速度和角速度上叠加小扰动。
            "root_vel_range": {
                "x": (-0.05, 0.05),  # m/s
                "y": (-0.05, 0.05),  # m/s
                "z": (-0.05, 0.05),  # m/s
                "roll": (-0.05, 0.05),  # rad/s
                "pitch": (-0.05, 0.05),  # rad/s
                "yaw": (-0.05, 0.05),  # rad/s
            },
            "joint_pos_range": (-0.2, 0.2),  # 关节位置附加 ±0.2rad 扰动。
            "joint_vel_range": (-0.5, 0.5),  # 关节速度附加 ±0.5rad/s 扰动。
        },
    )
    # reset：每个 episode 将所有执行器 Kp/Kd 分别乘以随机比例。
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),  # Kp 的80%~120%。
            "damping_distribution_params": (0.8, 1.2),  # Kd 的80%~120%。
            "operation": "scale",
        },
    )
    # reset：只随机脚踝关节内部摩擦和等效转子惯量。
    # randomize_ankle_dynamics = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*_joint"]),
    #         "friction_distribution_params": (0.7, 1.5),  # 标称摩擦的70%~150%。
    #         "armature_distribution_params": (0.7, 1.5),  # 标称 armature 的70%~150%。
    #         "operation": "scale",
    #     },
    # )
    # interval：训练过程中每隔3~6秒直接改变根速度，模拟外部推搡。
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),  # 每个环境独立采样下一次推搡间隔。
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),  # 根 x 线速度扰动，m/s。
                "y": (-0.5, 0.5),  # 根 y 线速度扰动，m/s。
                "z": (-0.05, 0.05),  # 根 z 线速度扰动，m/s。
                "roll": (-0.5, 0.5),  # roll 角速度扰动，rad/s。
                "pitch": (-0.5, 0.5),  # pitch 角速度扰动，rad/s。
                "yaw": (-0.5, 0.5),  # yaw 角速度扰动，rad/s。
            }
        },
    )


@configclass
class NoHim21DofRewardsCfg:
    """任务奖励配置；正 weight 为奖励，负 weight 为惩罚。"""

    # 机体朝向坐标系中的前后速度跟踪。std 越小，对 vx 误差越敏感。
    track_lin_vel_x_exp = RewTerm(func=mdp.track_lin_vel_x_yaw_frame_exp, weight=4.0, params={"std": 0.5})
    # 机体朝向坐标系中的侧向速度跟踪。
    track_lin_vel_y_exp = RewTerm(func=mdp.track_lin_vel_y_yaw_frame_exp, weight=3.0, params={"std": 0.5})
    # 世界坐标系绕 z 轴角速度跟踪，用于原地转弯和移动转弯。
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=3.0, params={"std": 0.5})
    # 纯转弯命令的附加奖励：跟踪 yaw 的同时抑制不需要的 xy 平移。
    # track_ang_vel_stand_exp = RewTerm(
    #     func=mdp.track_ang_vel_stand_world_exp,
    #     weight=5.0,
    #     # lin_vel_threshold 在该函数中是平移误差惩罚系数，不是速度触发阈值。
    #     params={"std": 0.5, "lin_vel_threshold": 2.0},
    # )
    # # 后退/后退侧移工况的额外角速度跟踪项，帮助后退时保持正确朝向变化。
    # track_ang_vel_backward_exp = RewTerm(
    #     func=mdp.track_ang_vel_backward_world_exp,
    #     weight=1.0,
    #     params={"std": 0.5},
    # )
    # # 任一速度分量跟踪误差超过0.3后开始产生惩罚，防止只摆姿态而不移动。
    # command_error_penalty = RewTerm(
    #     func=mdp.command_tracking_error_over_threshold,
    #     weight=-1.0,
    #     params={"threshold": 0.3},
    # )
    # # 奖励躯干保持竖直；std 控制对 roll/pitch 倾斜的容忍度。
    flat_orientation_exp = RewTerm(func=mdp.flat_orientation_exp, weight=0.5, params={"std": 0.5})
    # 奖励合理的双足接触模式：移动时偏向单脚支撑，站立时允许双脚支撑。
    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")},
    )
    # 接触力较大时，惩罚脚掌法向与接触力方向不一致，约束支撑脚着地姿态。
    feet_contact_orientation = RewTerm(
        func=mdp.feet_contact_orientation_error,
        weight=-6.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            # 仅在足部接触力超过50N时评估支撑脚方向。
            "force_threshold": 50.0,
        },
    )
    # # 摆动脚接近落地且向下运动时，惩罚脚掌四角的高度差，促使全脚掌近似水平落地。
    # pre_touchdown_feet_flat = RewTerm(
    #     func=mdp.pre_touchdown_feet_flat_error,
    #     weight=-1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #         # 四个点是踝 roll link 局部坐标系中的脚掌前后左右角点，单位m。
    #         "sole_points": (
    #             (0.105, 0.030, -0.050),
    #             (0.105, -0.030, -0.050),
    #             (-0.055, 0.030, -0.050),
    #             (-0.055, -0.030, -0.050),
    #         ),
    #         "near_ground_height": 0.08,  # 最低角点低于地面8cm才激活。
    #         "height_span_std": 0.008,  # 8mm高度差为误差尺度，越小要求越水平。
    #         "contact_force_threshold": 1.0,  # 接触力低于1N才视为尚未触地。
    #         "descending_velocity_threshold": -0.02,  # 脚部z速度低于-0.02m/s才视为下降。
    #         "ground_height": 0.0,  # 当前无限平面的世界z高度。
    #         "command_name": "base_velocity",
    #         # 平移或转向命令至少有一项超过对应阈值才激活，站立时返回0。
    #         "linear_command_threshold": 0.2,
    #         "angular_command_threshold": 0.2,
    #     },
    # )
    # # 摆动脚触地前惩罚过大的水平速度，降低向前擦地和瞬时制动冲击。
    # pre_touchdown_horizontal_speed = RewTerm(
    #     func=mdp.pre_touchdown_foot_horizontal_speed_error,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #         # 与上面的水平落脚奖励使用同一组脚掌角点。
    #         "sole_points": (
    #             (0.105, 0.030, -0.050),
    #             (0.105, -0.030, -0.050),
    #             (-0.055, 0.030, -0.050),
    #             (-0.055, -0.030, -0.050),
    #         ),
    #         "activation_height": 0.05,  # 脚掌最低点进入地面上方5cm后开始惩罚。
    #         "full_weight_height": 0.02,  # 低于2cm时使用完整惩罚权重。
    #         "horizontal_speed_tolerance": 0.20,  # 允许0.20m/s以内的水平落脚速度。
    #         "horizontal_speed_scale": 0.40,  # 超出容差后的归一化尺度，越小惩罚增长越快。
    #         "max_error": 4.0,  # 限制单脚归一化误差，避免极端冲击主导总奖励。
    #         "contact_force_threshold": 1.0,  # 只处理尚未有效接触的脚。
    #         "descending_velocity_threshold": -0.02,  # 只处理正在下降的脚。
    #         "ground_height": 0.0,
    #         "command_name": "base_velocity",
    #         # 只在明显移动/转向命令下激活，避免站立时鼓励脚部动作。
    #         "linear_command_threshold": 0.2,
    #         "angular_command_threshold": 0.2,
    #     },
    # )
    # # 足部接触地面期间惩罚水平滑动速度。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # 惩罚横向接触力相对法向力过大的冲击事件；它不是直接的竖直落地力惩罚。
    feet_impact = RewTerm(
        func=mdp.feet_impact,
        weight=-6.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")},
    )
    # 奖励移动时形成足够摆动时间；threshold=0.25s 是目标离地时长基准。
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 0.25,
        },
    )
    # 惩罚单脚法向接触力超过210N的部分，直接用于减轻落地过重。
    normal_feet_force = RewTerm(
        func=mdp.normal_feet_force,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 210,  # 超过该法向力阈值后才产生误差，单位N。
            "max_reward": 2000,  # 对函数原始误差封顶，限制异常接触造成的梯度尖峰。
        },
    )
    # 纯前后/站立命令下，惩罚双脚横向间距小于0.195m，避免双腿并拢或交叉。
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "threshold": 0.195,
        },
    )
    # 完整零命令站立时，惩罚两脚前后错位超过1cm，使站姿左右脚对齐。
    feet_normal_standing = RewTerm(
        func=mdp.feet_normal_standing,
        weight=-6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "threshold": 0.01,
        },
    )
    # 纯前后行走时约束髋yaw/roll和踝roll靠近默认姿态，减少横摆和侧翻投机动作。
    joint_deviation_leg = RewTerm(
        func=mdp.joint_deviation_l1_with_zero_flag,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_leg_yaw_joint", ".*_leg_roll_joint", ".*_ankle_roll_joint"],
            )
        },
    )
    # 完整零命令站立时，对机器人全部关节相对默认角度的L1偏差进行惩罚。
    joint_deviation_standing = RewTerm(
        func=mdp.joint_deviation_l1_when_standing,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 无xy平移命令时约束腰和上肢；纯原地转向也会受此项约束。
    joint_deviation_notwalk = RewTerm(
        func=mdp.joint_deviation_l1_when_notwalk,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_yaw_joint",
                    "l_arm_pitch_joint",
                    "l_arm_roll_joint",
                    "l_arm_yaw_joint",
                    "l_elbow_pitch_joint",
                    "r_arm_pitch_joint",
                    "r_arm_roll_joint",
                    "r_arm_yaw_joint",
                    "r_elbow_pitch_joint",
                ],
            )
        },
    )
    # 除左右脚踝roll link以外，任意机器人刚体接触力超过1N都视为非期望接触。
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[r"^(?!l_ankle_roll_link$)(?!r_ankle_roll_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )
    # 惩罚超过软关节位置限制的角度。
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    # 惩罚关节力矩乘关节速度的机械功率绝对值，降低能耗。
    energy = RewTerm(func=mdp.energy, weight=-5.0e-4)
    # 惩罚关节加速度平方，抑制高频振动；量纲大，因此权重很小。
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 惩罚连续两个控制周期动作之差的平方，使位置目标变化更平滑。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)


@configclass
class NoHim21DofTerminationsCfg:
    """Episode终止条件。"""

    # 达到episode时长后正常截断，不按失败终止处理。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 机体倾斜角超过0.6rad（约34.4度）时提前结束episode。
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6})


@configclass
class NoHim21DofActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=BUMI_21DOF_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class NoHim21DofObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        euler_angles = ObsTerm(func=mdp.euler_angles, noise=Unoise(n_min=-0.025, n_max=0.025))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = True
            self.history_length = 5
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        euler_angles = ObsTerm(func=mdp.euler_angles)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        feet_contact = ObsTerm(
            func=mdp.current_feet_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")},
        )
        payload = ObsTerm(func=mdp.payload, params={"asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_link"])})
        left_friction = ObsTerm(func=mdp.friction, params={"asset_cfg": SceneEntityCfg("robot", joint_names="l_ankle_roll_joint")})
        right_friction = ObsTerm(func=mdp.friction, params={"asset_cfg": SceneEntityCfg("robot", joint_names="r_ankle_roll_joint")})
        kp_params = ObsTerm(func=mdp.kp_params, params={"asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS)})
        kd_params = ObsTerm(func=mdp.kd_params, params={"asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS)})
        effort_limit = ObsTerm(func=mdp.effort_limit, params={"asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS)})

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class AmpObsCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.amp_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS)},
        )
        key_pos = ObsTerm(
            func=mdp.amp_key_pos,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FULL_BODY_BODIES)},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_vel = ObsTerm(
            func=mdp.amp_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS)},
        )
        feet_contact = ObsTerm(
            func=mdp.current_feet_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")},
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False
            self.flatten_history_dim = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    discriminator: AmpObsCfg = AmpObsCfg()


@configclass
class NoHim21DofEventCfg(NoHim21DofBaseEventCfg):
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS),
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )
    reset_robot_states = EventTerm(
        func=mdp.reset_root_state_amp,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=FULL_BODY_JOINTS),
            "reference_state_initialization": True,
            "prob_rsi": 0.8,
            "root_pos_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "root_vel_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "joint_pos_range": (-0.2, 0.2),
            "joint_vel_range": (-0.5, 0.5),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    randomize_actuator_effort_limit = EventTerm(
        func=mdp.randomize_actuator_effort_limit,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "effort_limit_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


@configclass
class NoHim21DofAmpEnvCfg(ManagerBasedRLEnvCfg):
    scene: NoHim21DofSceneCfg = NoHim21DofSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: NoHim21DofObservationsCfg = NoHim21DofObservationsCfg()
    actions: NoHim21DofActionsCfg = NoHim21DofActionsCfg()
    commands: NoHim21DofCommandsCfg = NoHim21DofCommandsCfg()
    events: NoHim21DofEventCfg = NoHim21DofEventCfg()
    rewards: NoHim21DofRewardsCfg = NoHim21DofRewardsCfg()
    terminations: NoHim21DofTerminationsCfg = NoHim21DofTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_sensor.update_period = self.sim.dt

        self.rand_arm = False
        self.dof_num = 21
        self.motion_loader_name = "MotionLoaderBumi"
        self.motion_files = sorted(glob.glob(f"{ASSET_DIR}/datasets/amp_data/bumi/bumi_walkrun/*.json"))
        if not self.motion_files:
            raise FileNotFoundError("No AMP motion files found in bumi/bumi_walkrun")
        self.reference_observation_horizon = 4
        self.num_preload_transitions = 100000
        self.observations.discriminator.history_length = self.reference_observation_horizon
