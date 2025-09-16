
# from gym.envs.registration import registry
#
# # re-register custom envs to replace gym's default ones
# def register(id, **kwargs):
#     if id in registry.env_specs:
#         registry.env_specs.pop(id)
#     return registry.register(id, **kwargs)
#
# register(
#     id='Reacher-v2',
#     entry_point='seqwm.envs.mujoco:ReacherEnv',
#     max_episode_steps=50,
#     reward_threshold=-3.75,
# )
#
# register(
#     id='Pusher-v2',
#     entry_point='seqwm.envs.mujoco:PusherEnv',
#     max_episode_steps=100,
#     reward_threshold=0.0,
# )
#
# register(
#     id='Thrower-v2',
#     entry_point='seqwm.envs.mujoco:ThrowerEnv',
#     max_episode_steps=100,
#     reward_threshold=0.0,
# )
#
# register(
#     id='Striker-v2',
#     entry_point='seqwm.envs.mujoco:StrikerEnv',
#     max_episode_steps=100,
#     reward_threshold=0.0,
# )
#
# register(
#     id='InvertedPendulum-v2',
#     entry_point='seqwm.envs.mujoco:InvertedPendulumEnv',
#     max_episode_steps=1000,
#     reward_threshold=950.0,
# )
#
# register(
#     id='InvertedDoublePendulum-v2',
#     entry_point='seqwm.envs.mujoco:InvertedDoublePendulumEnv',
#     max_episode_steps=1000,
#     reward_threshold=9100.0,
# )
#
# register(
#     id='HalfCheetah-v2',
#     entry_point='seqwm.envs.mujoco:HalfCheetahEnv',
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )
#
# register(
#     id='HalfCheetah-v3',
#     entry_point='seqwm.envs.mujoco.half_cheetah_v3:HalfCheetahEnv',
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )
#
# register(
#     id='Hopper-v2',
#     entry_point='seqwm.envs.mujoco:HopperEnv',
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
# )
#
# register(
#     id='Hopper-v3',
#     entry_point='seqwm.envs.mujoco.hopper_v3:HopperEnv',
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
# )
#
# register(
#     id='Swimmer-v2',
#     entry_point='seqwm.envs.mujoco:SwimmerEnv',
#     max_episode_steps=1000,
#     reward_threshold=360.0,
# )
#
# register(
#     id='Swimmer-v3',
#     entry_point='seqwm.envs.mujoco.swimmer_v3:SwimmerEnv',
#     max_episode_steps=1000,
#     reward_threshold=360.0,
# )
#
# register(
#     id='Walker2d-v2',
#     max_episode_steps=1000,
#     entry_point='seqwm.envs.mujoco:Walker2dEnv',
# )
#
# register(
#     id='Walker2d-v3',
#     max_episode_steps=1000,
#     entry_point='seqwm.envs.mujoco.walker2d_v3:Walker2dEnv',
# )
#
# register(
#     id='Ant-v2',
#     entry_point='seqwm.envs.mujoco:AntEnv',
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
# )
#
# register(
#     id='Ant-v3',
#     entry_point='seqwm.envs.mujoco.ant_v3:AntEnv',
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
# )
#
# register(
#     id='Humanoid-v2',
#     entry_point='seqwm.envs.mujoco:HumanoidEnv',
#     max_episode_steps=1000,
# )
#
# register(
#     id='Humanoid-v3',
#     entry_point='seqwm.envs.mujoco.humanoid_v3:HumanoidEnv',
#     max_episode_steps=1000,
# )
#
# register(
#     id='HumanoidStandup-v2',
#     entry_point='seqwm.envs.mujoco:HumanoidStandupEnv',
#     max_episode_steps=1000,
# )