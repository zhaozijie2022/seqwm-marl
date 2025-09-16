import numpy as np
from gym import utils
from . import mujoco_env

# action: [bthigh, bshin, bfoot, fthigh, fshin, ffoot]
# gemo_rgba 3-8： [bthigh, bshin, bfoot, fthigh, fshin, ffoot]

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        # body_names = self.model.body_names
        # self.body_idxes = {name: idx for idx, name in enumerate(body_names)}
        # self.xyz_index = {'x': 0, 'y': 1, 'z': 2}

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()

        reward_run = (xposafter - xposbefore) / self.dt  # runbwd

        # reward = reward_ctrl + reward_run
        reward = np.clip(reward_run, 0, 999) / 10
        info = {
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
            "origin_reward": reward_ctrl + reward_run,
        }
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
