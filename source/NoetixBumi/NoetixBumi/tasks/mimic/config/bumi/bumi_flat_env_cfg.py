from isaaclab.utils import configclass
from NoetixBumi.assets import ASSET_DIR
from NoetixBumi.assets.robots.bumi.bumi import BUMI_21DOF_CFG, BUMI_21DOF_ACTION_SCALE
from NoetixBumi.tasks.mimic.tracking_env_cfg import TrackingEnvCfg


@configclass
class BumiFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BUMI_21DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = BUMI_21DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.motion_file = ASSET_DIR + "/datasets/mimic_data/bumi/2.npz"

        self.commands.motion.ankle_dof_names = [
            'l_ankle_pitch_joint',
            'l_ankle_roll_joint',
            'r_ankle_pitch_joint',
            'r_ankle_roll_joint',
        ]
        self.commands.motion.body_names = [
            'base_link',
            'l_leg_pitch_link',
            # 'l_leg_roll_link',
            'l_leg_yaw_link',
            'l_knee_pitch_link',
            #'l_ankle_pitch_link',
            'l_ankle_roll_link',

            'r_leg_pitch_link',
            # 'r_leg_roll_link',
            'r_leg_yaw_link',
            'r_knee_pitch_link',
            # 'r_ankle_pitch_link',
            'r_ankle_roll_link',

            'waist_yaw_link',

            # 'l_arm_pitch_link',
            'l_arm_roll_link',
            # 'l_arm_yaw_link',
            'l_elbow_pitch_link',

            # 'r_arm_pitch_link',
            'r_arm_roll_link',
            # 'r_arm_yaw_link',
            'r_elbow_pitch_link']
