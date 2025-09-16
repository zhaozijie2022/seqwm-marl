import gym
from gym import spaces
import numpy
import torch
from copy import copy
from seqwm.envs.quadruped.mqe.envs.wrappers.empty_wrapper import EmptyWrapper

auto_yaw = True


class Go1GateWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(9 + self.num_agents * 8,), dtype=float)
        # self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(5 + self.num_agents * 8,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)

        self.target_reward_scale = 0.0
        self.x_pos_reward_scale = 1.0
        self.gate_reward_scale = 1.0
        self.success_reward_scale = 1.0
        self.lin_vel_x_reward_scale = 0
        self.contact_punishment_scale = -2.0
        self.lin_vel_y_punishment_scale = 0
        self.command_value_punishment_scale = 0

        self.approach_frame_punishment_scale = 0
        self.agent_distance_punishment_scale = -0.1
        self.agent_distance_threshold = 0.5
        self.frame_threshold = 0.04
        if auto_yaw:
            self.action_scale = torch.tensor(
                [[[[2, 2],
                   [0.5, 0.5],
                   [0.0, 0.0]]]]
            ).repeat(self.num_envs, self.num_agents, 1, 1).to(self.env.device)  # action_scale

            self.yaw_range = torch.tensor(
                [[[-1, 1]]]
            ).repeat(self.num_envs, self.num_agents, 1).to(self.env.device) * torch.pi / 180.0
        else:
            self.action_scale = torch.tensor(
                [[[[2, 2],
                   [0.5, 0.5],
                   [0.5, 0.5]]]]
            ).repeat(self.num_envs, self.num_agents, 1, 1).to(self.env.device)  # action_scale

    def _init_extras(self, obs):

        self.gate_pos = obs.env_info["gate_deviation"]
        self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)

        self.frame_left = self.gate_pos.reshape(-1, 2)
        self.frame_right = self.gate_pos.reshape(-1, 2)
        self.frame_left[:, 1] += self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.frame_right[:, 1] -= self.BarrierTrack_kwargs["gate"]["width"] / 2
        self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

        self.target_pos = torch.zeros_like(self.gate_pos, dtype=self.gate_pos.dtype, device=self.gate_pos.device)
        self.target_pos[:, :, 0] = self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] + \
                                   self.BarrierTrack_kwargs["plane"]["block_length"] / 2

        half_width = self.BarrierTrack_kwargs["track_width"] / 2
        y_offsets = torch.linspace(
            -half_width + half_width / self.num_agents,
            half_width - half_width / self.num_agents,
            steps=self.num_agents
        ).to(self.target_pos.device)
        self.target_pos[:, :, 1] = y_offsets

        # print("target", self.target_pos)

        return

    def _rescale_action(self, action):
        """
        Rescale action to match the action space of the environment.
        """

        action = torch.clip(action, -1, 1)  # action.shape (num_envs, num_agents, 3)
        scaled_action = action * (self.action_scale[..., 0] * (action <= 0) + self.action_scale[..., 1] * (action > 0))

        if auto_yaw:
            self.robot_yaw = (self.robot_yaw + torch.pi) % (2 * torch.pi) - torch.pi

            over_max = self.robot_yaw > self.yaw_range[..., 1]
            scaled_action[..., 2] = torch.where(
                over_max,
                - 0.5 * torch.ones_like(scaled_action[..., 2]),
                scaled_action[..., 2]
            )

            below_min = self.robot_yaw < self.yaw_range[..., 0]
            scaled_action[..., 2] = torch.where(
                below_min,
                + 0.5 * torch.ones_like(scaled_action[..., 2]),
                scaled_action[..., 2]
            )

        return scaled_action.reshape(-1, 3)

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        # base_pos = obs_buf.base_pos
        # base_rpy = obs_buf.base_rpy
        # base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        # obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), self.gate_pos], dim=2)
        obs = self.build_obs(obs_buf)
        return obs

    def step(self, action):
        action = self._rescale_action(action)
        obs_buf, _, termination, info = self.env.step(action)

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        base_pos = obs_buf.base_pos
        # base_rpy = obs_buf.base_rpy
        # base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        # obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), self.gate_pos], dim=2)

        obs = self.build_obs(obs_buf)

        # region compute reward
        reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)

        if self.target_reward_scale != 0:
            self.target_pos = self.target_pos.view(-1, 2)  # 25/8/27 csq debug
            distance_to_taget = torch.norm(base_pos[:, :2] - self.target_pos, p=2, dim=1)

            if not hasattr(self, "last_distance_to_taget"):
                self.last_distance_to_taget = copy(distance_to_taget)

            target_reward = (self.last_distance_to_taget - distance_to_taget)
            if self.gate_reward_scale > 0:
                unpassed_mask = base_pos[:, 0] < self.gate_distance + 0.25
                target_reward[unpassed_mask] = 0
            target_reward = target_reward.reshape(self.num_envs, -1).sum(dim=1, keepdim=True)

            target_reward[self.env.reset_ids] = 0
            target_reward *= self.target_reward_scale
            reward += target_reward.repeat(1, self.env.num_agents)

            self.last_distance_to_taget = copy(distance_to_taget)

        if self.gate_reward_scale != 0:
            gate_pos = self.gate_pos.view(-1, 2).clone()
            gate_pos[:, 0] += 0.25
            distance_to_gate = torch.norm(base_pos[:, :2] - gate_pos, p=2, dim=1)  # (n_envs * n_agents, 1)

            if not hasattr(self, "last_distance_to_gate"):
                self.last_distance_to_gate = copy(distance_to_gate)

            gate_reward = (self.last_distance_to_gate - distance_to_gate)

            passed_mask = base_pos[:, 0] > self.gate_distance + 0.25
            gate_reward[passed_mask] = 0
            gate_reward = gate_reward.reshape(self.num_envs, -1).sum(dim=1, keepdim=True)

            gate_reward[self.env.reset_ids] = 0
            gate_reward *= self.gate_reward_scale
            reward += gate_reward.repeat(1, self.env.num_agents)

            self.last_distance_to_gate = copy(distance_to_gate)

        if self.x_pos_reward_scale != 0:
            x_pos = base_pos[:, 0]
            if not hasattr(self, "last_x_pos"):
                self.last_x_pos = copy(x_pos)
            x_pos_reward = (x_pos - self.last_x_pos).reshape(self.num_envs, -1).sum(dim=1, keepdim=True)

            x_pos_reward[self.env.reset_ids] = 0
            x_pos_reward *= self.x_pos_reward_scale
            reward += x_pos_reward.repeat(1, self.num_agents)
            self.last_x_pos = copy(x_pos)

        if self.contact_punishment_scale != 0:
            collide_reward = self.contact_punishment_scale * self.env.collide_buf
            reward += collide_reward.unsqueeze(1).repeat(1, self.num_agents)

        if self.success_reward_scale != 0:
            success_reward = torch.zeros((self.num_envs * self.num_agents), device=self.env.device)
            passed_mask = base_pos[:, 0] > self.gate_distance + 0.25
            all_passed_mask = passed_mask.reshape(self.num_envs, self.num_agents).all(dim=1, keepdim=True)
            all_passed_mask = all_passed_mask.repeat(1, self.num_agents)  # Tensor [num_env, num_agents]
            success_reward[all_passed_mask.reshape(-1)] = self.success_reward_scale
            reward += success_reward.reshape([self.num_envs, self.num_agents])

        # approach frame punishment
        if self.approach_frame_punishment_scale != 0:
            dis_to_left_frame = ((base_pos[:, :2] - self.frame_left) ** 2).sum(dim=1).reshape(self.num_envs, -1)
            dis_to_right_frame = ((base_pos[:, :2] - self.frame_right) ** 2).sum(dim=1).reshape(self.num_envs, -1)

            approach_left = self.approach_frame_punishment_scale / (
                        dis_to_left_frame[dis_to_left_frame < self.frame_threshold] + self.frame_threshold)
            approach_right = self.approach_frame_punishment_scale / (
                        dis_to_right_frame[dis_to_left_frame < self.frame_threshold] + self.frame_threshold)
            reward[dis_to_left_frame < self.frame_threshold] += approach_left
            reward[dis_to_right_frame < self.frame_threshold] += approach_right

        # agent distance punishment
        if self.agent_distance_punishment_scale != 0:
            base_pos_2d = base_pos[:, :2].reshape(self.num_envs, self.num_agents, 2)
            agent_dis = torch.cdist(base_pos_2d, base_pos_2d, p=2) ** 2
            mask = torch.eye(self.num_agents, device=self.env.device).bool()
            agent_dis.masked_fill_(mask.unsqueeze(0), float('inf'))
            min_agent_dis, _ = agent_dis.min(dim=2)

            agent_distance_punishment = self.agent_distance_punishment_scale / min_agent_dis[min_agent_dis < self.agent_distance_threshold]

            reward[min_agent_dis < self.agent_distance_threshold] += agent_distance_punishment

        # command lin_vel.y punishment
        if self.lin_vel_y_punishment_scale != 0:
            v_y_punishment = self.lin_vel_y_punishment_scale * action[:, :, 1] ** 2
            reward += v_y_punishment

        # command value punishment
        if self.command_value_punishment_scale != 0:
            command_value_punishment = self.command_value_punishment_scale * torch.clip(action ** 2 - 1, 0, 1).sum(dim=2)
            reward += command_value_punishment

        # lin_vel.x reward
        if self.lin_vel_x_reward_scale != 0:
            v_x_reward = self.lin_vel_x_reward_scale * obs_buf.lin_vel[:, 0].reshape(self.num_envs, self.num_agents)
            reward += v_x_reward

        reward = reward.sum(dim=1).unsqueeze(1).repeat(1, self.num_agents)
        # obs, reward = 0, 0

        passed_mask = base_pos[:, 0] > self.gate_distance + 0.25
        all_passed_mask = passed_mask.reshape(self.num_envs, self.num_agents).all(dim=1, keepdim=True)
        return_info = [
            [{"success": bool(all_passed_mask[env_id])}] * self.num_agents for env_id in range(self.num_envs)
        ]

        return obs, reward, termination, return_info

    def build_obs(self, obs_buf):
        data_2d = self.get_things(obs_buf)
        self.data_2d = data_2d

        other_pos_reshaped = data_2d["other_2d"].reshape(self.num_envs, self.num_agents, self.num_agents - 1, 3)
        other_xyyaw = []
        other_dist = []
        other_sin_yaw = []
        other_cos_yaw = []
        success_flag = []

        for i in range(self.num_agents - 1):
            other_single = other_pos_reshaped[:, :, i, :]
            trans_single = self.frame_transform_2d(data_2d["self_2d"], other_single)
            other_xyyaw.append(trans_single)
            other_dist.append(torch.norm(trans_single[:, :, :2], dim=-1, keepdim=True))
            other_sin_yaw.append(torch.sin(trans_single[..., 2:3]))
            other_cos_yaw.append(torch.cos(trans_single[..., 2:3]))
            success_flag.append(
                (other_single[..., 0] > self.gate_distance.reshape(self.num_envs, self.num_agents) + 0.25).float().unsqueeze(-1))

        success_flag.append(
            (data_2d["self_2d"][..., 0] > self.gate_distance.reshape(self.num_envs, self.num_agents) + 0.25).float().unsqueeze(-1))

        other_xyyaw_ = torch.cat(other_xyyaw, dim=-1)
        other_dist_ = torch.cat(other_dist, dim=-1)
        other_sin_yaw_ = torch.cat(other_sin_yaw, dim=-1)
        other_cos_yaw_ = torch.cat(other_cos_yaw, dim=-1)
        success_flag_ = torch.cat(success_flag, dim=-1)

        gate_2d = self.frame_transform_2d(data_2d["self_2d"], data_2d["gate_2d"])
        behind_gate_2d = self.frame_transform_2d(data_2d["self_2d"], data_2d["behind_gate_2d"])
        left_frame_2d = self.frame_transform_2d(data_2d["self_2d"], data_2d["left_frame_2d"])
        right_frame_2d = self.frame_transform_2d(data_2d["self_2d"], data_2d["right_frame_2d"])
        target_2d = self.frame_transform_2d(data_2d["self_2d"], data_2d["target_2d"])

        self.robot_yaw = data_2d["self_2d"][..., 2]  # (num_envs, num_agents, 1)

        agent_id = self.obs_ids

        gate_xy = gate_2d[..., :2]
        gate_dist = torch.norm(gate_2d[..., :2], dim=-1, keepdim=True)

        behind_gate_xy = behind_gate_2d[..., :2]
        behind_gate_dist = torch.norm(behind_gate_2d[..., :2], dim=-1, keepdim=True)

        left_frame_xy = left_frame_2d[..., :2]
        left_frame_dist = torch.norm(left_frame_2d[..., :2], dim=-1, keepdim=True)

        right_frame_xy = right_frame_2d[..., :2]
        right_frame_dist = torch.norm(right_frame_2d[..., :2], dim=-1, keepdim=True)

        target_xy = target_2d[..., :2]
        target_dist = torch.norm(target_2d[..., :2], dim=-1, keepdim=True)

        final_obs = torch.cat([
            agent_id,  # (num_envs, num_agents, num_agents)
            other_xyyaw_,  # (num_envs, num_agents, 3 * (num_agents - 1))
            other_dist_,  # (num_envs, num_agents, num_agents - 1)
            other_sin_yaw_,  # (num_envs, num_agents, num_agents - 1)
            other_cos_yaw_,  # (num_envs, num_agents, num_agents - 1)
            gate_xy,
            gate_dist,
            left_frame_xy,
            left_frame_dist,
            right_frame_xy,
            right_frame_dist,
            target_xy,
            target_dist,
            behind_gate_xy,
            behind_gate_dist,
            success_flag_  # (num_envs, num_agents, num_agents - 1)
        ], dim=-1)

        # print(final_obs.shape)
        return final_obs

    def get_things(self, obs_buf):
        self_xyz = obs_buf.base_pos.reshape(self.num_envs, self.num_agents, 3)
        self_rpy = obs_buf.base_rpy.reshape(self.num_envs, self.num_agents, 3)
        self_pos = torch.cat([self_xyz[..., :2], self_rpy[..., 2:]], dim=-1)

        mask = ~torch.eye(self.num_agents, dtype=torch.bool, device=self_pos.device)
        expanded_self_pos = self_pos.unsqueeze(1).expand(-1, self.num_agents, -1, -1)
        other_pos = expanded_self_pos[:, mask].reshape(self.num_envs, self.num_agents, 3 * (self.num_agents - 1))

        gate_pos = self.gate_pos.reshape(self.num_envs, self.num_agents, 2)
        gate_pos = torch.cat([gate_pos, torch.zeros_like(gate_pos[..., :1])], dim=-1)

        behind_gate_pos = gate_pos.clone()
        behind_gate_pos[..., 0] += 0.25

        left_frame_pos = self.frame_left.reshape(self.num_envs, self.num_agents, 2)
        left_frame_pos = torch.cat([left_frame_pos, torch.zeros_like(left_frame_pos[..., :1])], dim=-1)

        right_frame_pos = self.frame_right.reshape(self.num_envs, self.num_agents, 2)
        right_frame_pos = torch.cat([right_frame_pos, torch.zeros_like(right_frame_pos[..., :1])], dim=-1)

        target_pos = self.target_pos.reshape(self.num_envs, self.num_agents, 2)
        target_pos = torch.cat([target_pos, torch.zeros_like(target_pos[..., :1])], dim=-1)

        data = {
            "self_2d": self_pos,
            "other_2d": other_pos,
            "gate_2d": gate_pos,
            "behind_gate_2d": behind_gate_pos,
            "left_frame_2d": left_frame_pos,
            "right_frame_2d": right_frame_pos,
            "target_2d": target_pos,
        }
        return data

    @staticmethod
    def frame_transform_2d(self_pos, object_pos):
        delta = object_pos[..., :2] - self_pos[..., :2]
        yaw = self_pos[..., 2]

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        x_rel = cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1]
        y_rel = -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1]
        yaw_rel = object_pos[..., 2] - self_pos[..., 2]

        return torch.stack([x_rel, y_rel, yaw_rel], dim=-1)
