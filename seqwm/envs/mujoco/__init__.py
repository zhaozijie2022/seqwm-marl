from .mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from .ant import AntEnv
from .half_cheetah import HalfCheetahEnv
from .hopper import HopperEnv
from .walker2d import Walker2dEnv
from .humanoid import HumanoidEnv
from .inverted_pendulum import InvertedPendulumEnv
from .inverted_double_pendulum import InvertedDoublePendulumEnv
from .reacher import ReacherEnv
from .swimmer import SwimmerEnv
from .humanoidstandup import HumanoidStandupEnv
from .pusher import PusherEnv
from .thrower import ThrowerEnv
from .striker import StrikerEnv
