import gym.spaces
import torch
from seqwm.envs.quadruped.mqe.envs.utils import make_mqe_env, custom_cfg
from seqwm.envs.quadruped.mqe.utils import get_args
from seqwm.envs.dexhands.dexhands_env import _t2n



class QuadrupedEnv:
    def __init__(self, env_args):
        self.env_args = env_args
        init_args = get_args()
        init_args.num_envs = env_args["n_threads"]
        init_args.seed = env_args.get("seed", 0)
        init_args.headless = env_args.get("headless", True)
        init_args.record_video = env_args.get("record_video", False)
        self.env, self.env_cfg = make_mqe_env(
            env_name=env_args["task"],
            args=init_args,
            custom_cfg=custom_cfg(init_args),
        )
        self.n_envs = self.env.num_envs
        self.n_agents = self.env.num_agents
        self.observation_space = [
            self.env.observation_space for _ in range(self.n_agents)
        ]
        self.action_space = [
            self.env.action_space for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            gym.spaces.Box(low=-10, high=10, shape=(self.env.observation_space.shape[0] * self.n_agents, ))
            for _ in range(self.n_agents)
        ]

    def step(self, actions):
        actions = torch.tensor(actions, device=self.env.device)
        # actions dim: [num_envs, num_agents, action_dims]
        obs_all, reward_all, done_all, info_all = self.env.step(actions)
        # out dims: [num_envs, num_agents, xxx_dims]
        state_all = obs_all.view(self.n_envs, -1).unsqueeze(1).repeat(1, self.n_agents, 1)
        return (
            _t2n(obs_all),
            _t2n(state_all),
            _t2n(reward_all.view(self.n_envs, self.n_agents, 1)),
            _t2n(done_all.unsqueeze(-1).repeat(1, self.n_agents)),
            [[{}, {}]] * self.n_envs,
            [None] * self.env_args["n_threads"],  # available_actions
        )

    def reset(self):
        obs = self.env.reset()
        s_obs = obs.view(self.n_envs, -1).unsqueeze(1).repeat(1, self.n_agents, 1)
        return _t2n(obs), _t2n(s_obs), [None] * self.env_args["n_threads"]

    def close(self):
        pass









































