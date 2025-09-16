# environments
from seqwm.envs.quadruped.mqe.envs.field.legged_robot_field import LeggedRobotField
from seqwm.envs.quadruped.mqe.envs.go1.go1 import Go1
from seqwm.envs.quadruped.mqe.envs.npc.go1_sheep import Go1Sheep
from seqwm.envs.quadruped.mqe.envs.npc.go1_object import Go1Object

# configs
from seqwm.envs.quadruped.mqe.envs.field.legged_robot_field_config import LeggedRobotFieldCfg
from seqwm.envs.quadruped.mqe.envs.configs.go1_gate_config import Go1GateCfg
from seqwm.envs.quadruped.mqe.envs.configs.go1_sheep_config import SingleSheepCfg

# wrappers
from seqwm.envs.quadruped.mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from seqwm.envs.quadruped.mqe.envs.wrappers.go1_gate_wrapper import Go1GateWrapper
from seqwm.envs.quadruped.mqe.envs.wrappers.go1_pushbox_wrapper import Go1PushboxWrapper
from seqwm.envs.quadruped.mqe.envs.wrappers.go1_sheep_wrapper import Go1SheepWrapper

from seqwm.envs.quadruped.mqe.utils import get_args, make_env

from typing import Tuple

ENV_DICT = {
    "go1gate": {
        "class": Go1,
        "config": Go1GateCfg,
        "wrapper": Go1GateWrapper
    },
    "go1sheep-easy": {
        "class": Go1Sheep,
        "config": SingleSheepCfg,
        "wrapper": Go1SheepWrapper
    },
    "go1pushbox": {
        "class": Go1Object,
        "config": Go1PushboxCfg,
        "wrapper": Go1PushboxWrapper
    },
}


def make_mqe_env(env_name: str, args=None, custom_cfg=None) -> Tuple[LeggedRobotField, LeggedRobotFieldCfg]:
    
    env_dict = ENV_DICT[env_name]

    if callable(custom_cfg):
        env_dict["config"] = custom_cfg(env_dict["config"])

    env, env_cfg = make_env(env_dict["class"], env_dict["config"], args)
    env = env_dict["wrapper"](env)

    return env, env_cfg

def custom_cfg(args):

    def fn(cfg:LeggedRobotFieldCfg):
        
        if getattr(args, "num_envs", None) is not None:
            cfg.env.num_envs = args.num_envs
        
        cfg.env.record_video = args.record_video

        return cfg
    
    return fn