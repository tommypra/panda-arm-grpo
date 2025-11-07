import os
from panda_mujoco_gym.envs.panda_custom_env import FrankaCustomEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "reach_new.xml")


class FrankaReachEnv(FrankaCustomEnv):
    def __init__(
        self,
        reward_type,
        **kwargs,
    ):
        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=True,
            distance_threshold=0.05,
            goal_xy_range=0.8,
            goal_x_offset=0.3,
            goal_z_range=0.6,
            **kwargs,
        )
