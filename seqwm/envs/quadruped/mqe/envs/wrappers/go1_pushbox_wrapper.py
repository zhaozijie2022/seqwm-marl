from copy import copy
from copy import deepcopy
from gym import spaces
from seqwm.envs.quadruped.mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from isaacgym.torch_utils import *
import torch

USE_MA_PUSH = True
BOX_LENGTH = 1.0

class Go1PushboxWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(33,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor(
            [[[[2.0, 2.0],
               [0.5, 0.5],
               [0.5, 0.5]]]]
        ).repeat(self.num_envs, self.num_agents, 1, 1).to(self.env.device)  # action_scale

        self.final_threshold = 0.3

        self.target_reward_scale = 0.00325
        self.approach_reward_scale = 0.00250
        self.toward_reward_scale = 0.00100
        self.collision_punishment_scale = -0.00250
        self.push_reward_scale = 0.00150
        self.ocb_reward_scale = 0.00750
        self.reach_target_reward_scale = 50
        self.exception_punishment_scale = -5
        self.success_reward_scale = 0

        self.last_box_state = None
        self.last_box_pos = None

    def reset(self):
        obs_buf = self.env.reset()
        data_3d = self.get_things_3d(obs_buf)
        data_2d = self.build_3d_to_2d(data_3d)
        obs = self.build_obs(data_2d)
        return obs

    def _rescale_action(self, action):
        """
        Rescale action to match the action space of the environment.
        """
        action = torch.clip(action, -1, 1)  # action.shape (num_envs, num_agents, 3)
        scaled_action = action * (
                self.action_scale[..., 0] * (action <= 0)
                + self.action_scale[..., 1] * (action > 0)
        )
        return scaled_action

    def step(self, action):
        # action
        action = self._rescale_action(action)  # (num_envs, num_agents, 3)
        action[self.check_final_success()] = 0
        action = action.reshape(-1, 3)  # (num_envs * num_agents, 3)

        # step
        obs_buf, _, termination, info = self.env.step(action)

        # build_obs
        data_3d = self.get_things_3d(obs_buf)
        data_2d = self.build_3d_to_2d(data_3d)
        obs = self.build_obs(data_2d)
        # check value exception
        self.value_exception_buf = torch.isnan(obs).any(dim=2).any(dim=1) \
                                   | torch.isinf(obs).any(dim=2).any(dim=1)
        # remove nan and inf in obs and reward
        obs[torch.isnan(obs) | torch.isinf(obs)] = 0

        # reward
        reward, return_info = self.get_reward(obs_buf=obs_buf)

        return obs, reward, termination, return_info

    def build_obs(self, data_2d):
        self.data_2d = data_2d

        other_2d = self.frame_transform_2d(data_2d["self_2d"], data_2d["other_2d"], )
        box_2d = self.frame_transform_2d(data_2d["self_2d"], data_2d["box_2d"])
        target_2d = self.frame_transform_2d(data_2d["self_2d"], data_2d["target_2d"])

        agent_id = self.obs_ids

        other_xyyaw = other_2d[:, :, :3]  # (num_envs, num_agents, 3)
        other_dist = torch.norm(other_2d[:, :, :2], dim=-1, keepdim=True)  # (num_envs, num_agents, 1)
        other_sin_yaw = torch.sin(other_2d[:, :, 2:3])  # (num_envs, num_agents, 1)
        other_cos_yaw = torch.cos(other_2d[:, :, 2:3])  # (num_envs, num_agents, 1)

        box_xyyaw = box_2d[:, :, :3]  # (num_envs, num_agents, 3)
        box_dist = torch.norm(box_2d[:, :, :2], dim=-1, keepdim=True)  # (num_envs, num_agents, 1)
        box_sin_yaw = torch.sin(box_2d[:, :, 2:3])  # (num_envs, num_agents, 1)
        box_cos_yaw = torch.cos(box_2d[:, :, 2:3])  # (num_envs, num_agents, 1)

        target_xy = target_2d[:, :, :2]  # (num_envs, num_agents, 2)
        target_dist = torch.norm(target_2d[:, :, :2], dim=-1, keepdim=True)  # (num_envs, num_agents, 1)

        target_box_xy = target_xy - box_2d[:, :, :2]  # (num_envs, num_agents, 2)
        target_box_dist = torch.norm(target_box_xy, dim=-1, keepdim=True)

        box_tops = self.get_box_tops(box_2d[:, :, :2], box_2d[:, :, 2:3], BOX_LENGTH)  # (num_envs, num_agents, 4, 2)
        box_tops_dist = torch.norm(box_tops, dim=-1, keepdim=True)  # (num_envs, num_agents, 4, 1)
        box_tops_2d = torch.cat([box_tops, box_tops_dist], dim=-1)  # (num_envs, num_agents, 4, 3)
        box_tops_2d_flat = box_tops_2d.reshape(self.num_envs, self.num_agents, -1)  # (num_envs, num_agents, 12)

        threshold = torch.tensor(
            [[[self.final_threshold]] * self.num_agents] * self.num_envs,
            dtype=torch.float32, device=self.env.device
        )

        obs = torch.cat([
            agent_id,  # (num_envs, num_agents, 2)
            other_xyyaw,  # (num_envs, num_agents, 3) [x, y, yaw]
            other_dist,  # (num_envs, num_agents, 1) [distance]
            other_sin_yaw,  # (num_envs, num_agents, 1) [sin_yaw]
            other_cos_yaw,  # (num_envs, num_agents, 1) [cos_yaw]
            box_xyyaw,  # (num_envs, num_agents, 3) [x, y, yaw]
            box_dist,  # (num_envs, num_agents, 1) [distance]
            box_sin_yaw,  # (num_envs, num_agents, 1) [sin_yaw]
            box_cos_yaw,  # (num_envs, num_agents, 1) [cos_yaw]
            target_xy,  # (num_envs, num_agents, 2) [x, y]
            target_dist,  # (num_envs, num_agents, 1) [distance]
            target_box_xy,  # (num_envs, num_agents, 2) [x, y]
            target_box_dist,  # (num_envs, num_agents, 1) [distance]
            box_tops_2d_flat,  # (num_envs, num_agents, 12) [x1, y1, d1, x2, y2, d2, x3, y3, d3, x4, y4, d4]
            threshold,  # (num_envs, num_agents, 1) [threshold]
            # success_flag
        ], dim=-1)  # (num_envs, num_agents, 31)

        return obs

    def get_things_3d(self, obs_buf):
        # self-xyz
        self_pos = obs_buf.base_pos.reshape([self.num_envs, self.num_agents, -1])
        self_rpy = obs_buf.base_rpy.reshape([self.num_envs, self.num_agents, -1])

        # other-xyz
        other_pos = torch.flip(self_pos, dims=[1])  # (num_envs, num_agents, 3)
        other_rpy = torch.flip(self_rpy, dims=[1])

        # box-xyz
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:, 0, :] - self.env.env_origins
        # box_pos[torch.isnan(box_pos) | torch.isinf(box_pos)] = 0
        box_pos = box_pos.reshape(self.num_envs, 1, -1).repeat(1, self.num_agents, 1)
        # box-rpy
        box_quaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_quaternion), dim=1)
        # box_rpy[torch.isnan(box_rpy) | torch.isinf(box_rpy)] = 0
        box_rpy = box_rpy.reshape(self.num_envs, 1, -1).repeat(1, self.num_agents, 1)

        # target-xyz
        target_pos = npc_pos[:, 1, :] - self.env.env_origins
        # target_pos[torch.isnan(target_pos) | torch.isinf(target_pos)] = 0
        target_pos = target_pos.reshape(self.num_envs, 1, -1).repeat(1, self.num_agents, 1)
        # target-rpy
        target_quaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1, 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_quaternion), dim=1)
        # target_rpy[torch.isnan(target_rpy) | torch.isinf(target_rpy)] = 0
        target_rpy = target_rpy.reshape(self.num_envs, 1, -1).repeat(1, self.num_agents, 1)

        return {
            "self_pos": self_pos,  # (num_envs, num_agents, 3)
            "self_rpy": self_rpy,  # (num_envs, num_agents, 3)
            "other_pos": other_pos,  # (num_envs, num_agents, 3)
            "other_rpy": other_rpy,  # (num_envs, num_agents, 3)
            "box_pos": box_pos,  # (num_envs, num_agents, 3)
            "box_rpy": box_rpy,  # (num_envs, num_agents, 3)
            "target_pos": target_pos,  # (num_envs, num_agents, 3)
            "target_rpy": target_rpy,  # (num_envs, num_agents, 3)
        }

    @staticmethod
    def build_3d_to_2d(data_3d):
        data_2d = {
            "self_2d": torch.cat([
                data_3d["self_pos"][..., :2],
                data_3d["self_rpy"][..., 2:3]
            ], dim=2),  # (num_envs, num_agents, 3)
            "other_2d": torch.cat([
                data_3d["other_pos"][..., :2],
                data_3d["other_rpy"][..., 2:3]
            ], dim=2),  # (num_envs, num_agents, 3)
            "box_2d": torch.cat([
                data_3d["box_pos"][..., :2],
                data_3d["box_rpy"][..., 2:3]
            ], dim=2),  # (num_envs, num_agents, 3)
            "target_2d": torch.cat([
                data_3d["target_pos"][..., :2],
                data_3d["target_rpy"][..., 2:3]
            ], dim=2),  # (num_envs, num_agents, 3)
        }
        return data_2d

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

    @staticmethod
    def get_box_tops(center: torch.Tensor, theta: torch.Tensor, length=None):
        if length is None:
            length = BOX_LENGTH
        N, M = center.shape[:2]
        half_L = length / 2.0

        local_pts = torch.tensor([
            [-half_L, -half_L],
            [half_L, -half_L],
            [half_L, half_L],
            [-half_L, half_L]
        ], dtype=center.dtype, device=center.device)  # [4, 2]

        center_flat = center.reshape(-1, 2)  # [N*M, 2]
        theta_flat = theta.reshape(-1, 1).squeeze()  # [N*M, 1]
        local_pts = local_pts.unsqueeze(0).repeat(N * M, 1, 1)  # [N*M, 4, 2]

        cos = torch.cos(theta_flat)
        sin = torch.sin(theta_flat)
        R = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1)
        ], dim=-2)  # [N*M, 2, 2]

        rotated_pts = torch.bmm(local_pts, R.transpose(1, 2))  # [N*M, 4, 2]
        vertices = rotated_pts + center_flat.unsqueeze(1)  # [N*M, 4, 2]
        return vertices

    def calc_normal_vector_for_obc_reward(self, vertex_list, pos_tensor):
        if vertex_list is None:
            vertex_list = [
                [-BOX_LENGTH / 2, -BOX_LENGTH / 2],
                [BOX_LENGTH / 2, -BOX_LENGTH / 2],
                [BOX_LENGTH / 2, BOX_LENGTH / 2],
                [-BOX_LENGTH / 2, BOX_LENGTH / 2]
            ]
        pos_tensor = pos_tensor.to(self.device)
        vertices = torch.tensor(vertex_list, device=self.device).float()
        num_vertices = vertices.shape[0]

        edges = torch.roll(vertices, -1, dims=0) - vertices
        vp = pos_tensor[:, None, :] - vertices[None, :, :]

        edges_expanded = edges[None, :, :].repeat(pos_tensor.shape[0], 1, 1)
        edge_lengths = torch.norm(edges_expanded, dim=2, keepdim=True)
        edge_unit = edges_expanded / edge_lengths
        edge_normals = torch.stack([-edge_unit[:, :, 1], edge_unit[:, :, 0]], dim=2)

        cross_prod = torch.abs(vp[:, :, 0] * edge_unit[:, :, 1] - vp[:, :, 1] * edge_unit[:, :, 0])
        dot_product1 = (vp * edges_expanded).sum(dim=2)
        dot_product2 = (torch.roll(vp, -1, dims=1) * edges_expanded).sum(dim=2)

        on_segment = (dot_product1 >= 0) & (dot_product2 <= 0)
        dist_to_line = torch.where(on_segment, cross_prod, torch.tensor(float('inf'), device=self.device))

        dist_to_vertex1 = torch.norm(vp, dim=2)
        dist_to_vertex2 = torch.norm(pos_tensor[:, None, :] - torch.roll(vertices, -1, dims=0)[None, :, :], dim=2)

        min_dist_each_edge, indices = torch.min(torch.stack([dist_to_line, dist_to_vertex1, dist_to_vertex2], dim=-1),
                                                dim=2)
        min_dist, indices = torch.min(min_dist_each_edge, dim=1)
        selected_normals = edge_normals[0][indices]

        return selected_normals

    def check_final_success(self):
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_target_xy = npc_pos[:, 0, :2] - npc_pos[:, 1, :2]
        box_target_dist = torch.norm(box_target_xy, dim=1)  # (num_envs,)
        final_success_mask = box_target_dist < self.final_threshold  # (num_envs,)
        return final_success_mask

    def get_reward(self, obs_buf):
        # calculate reward
        def rotation_matrix_2D(theta):
            theta = theta.float()
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotation_matrices = torch.stack([
                torch.stack([cos_theta, -sin_theta], dim=1),
                torch.stack([sin_theta, cos_theta], dim=1)
            ], dim=1)
            return rotation_matrices

        box_state = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0]
        target_state = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1]
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:, 0, :] - self.env.env_origins
        target_pos = npc_pos[:, 1, :] - self.env.env_origins
        box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
        target_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1, 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_qyaternion), dim=1)

        base_pos = obs_buf.base_pos  # (env_num, agent_num, 3)
        base_rpy = obs_buf.base_rpy  # (env_num, agent_num, 3)
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # occlude nan or inf
        box_pos[torch.isnan(box_pos)] = 0
        box_pos[torch.isinf(box_pos)] = 0
        target_pos[torch.isnan(target_pos)] = 0
        target_pos[torch.isinf(target_pos)] = 0
        base_pos[torch.isnan(base_pos)] = 0
        base_pos[torch.isinf(base_pos)] = 0
        box_rpy[torch.isnan(box_rpy)] = 0
        box_rpy[torch.isinf(box_rpy)] = 0

        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device) - 0.1

        # calculate reach target reward and set finish task termination
        if self.reach_target_reward_scale != 0:  # 10
            reward[self.finished_buf, :] += self.reach_target_reward_scale

        b2t_distance = torch.norm((box_pos[:, :2] - target_pos[:, :2]), p=2, dim=1)  # (num_envs,)
        success_mask = b2t_distance < self.success_threshold
        return_info = [
            [{"success": bool(self.finished_buf[env_id])}] * self.env.num_agents
            for env_id in range(self.num_envs)
        ]  # List[num_env, List[num_agents, bool]]
        if self.success_reward_scale != 0:
            success_reward = self.success_reward_scale / (b2t_distance + self.final_threshold)
            reward[:, :] += success_reward.unsqueeze(1).repeat(1, self.num_agents)

        # calculate distance from current_box_pos to target_box_pos reward
        if self.target_reward_scale != 0:
            if self.last_box_state is None:
                self.last_box_state = copy(box_state)
            if self.last_box_pos is None:
                self.last_box_pos = copy(box_pos)

            # past_distance = self.env.dist_calculator.cal_dist(self.last_box_state, target_state)
            # distance = self.env.dist_calculator.cal_dist(box_state, target_state)
            past_distance = torch.norm(target_pos[:, :2] - self.last_box_pos[:, :2], p=2, dim=1)
            distance = torch.norm(box_pos[:, :2] - target_pos[:, :2], p=2, dim=1)  # (num_envs,)
            distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
            reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)

        # calculate distance from each robot to box reward
        if self.approach_reward_scale != 0:
            reward_logger = []
            for i in range(self.num_agents):
                distance = torch.norm(box_pos - base_pos[:, i, :], dim=1, keepdim=True)
                _mask = distance > (BOX_LENGTH / 2 + 0.30)
                distance_reward = torch.zeros_like(distance)
                distance_reward[_mask] = -((distance[_mask] + 0.5) ** 2) * self.approach_reward_scale
                # distance_reward = -((distance + 0.5) ** 2) * self.approach_reward_scale
                reward_logger.append(torch.sum(distance_reward).cpu())
                reward[:, i] += distance_reward.squeeze(-1)

        if self.toward_reward_scale != 0:
            reward_logger = []
            for i in range(self.num_agents):
                rel_pos = box_pos[:, :2] - base_pos[:, i, :2]
                distance = torch.norm(rel_pos, dim=1, keepdim=True)
                ideal_toward = rel_pos / (distance + 1e-6)
                self_toward = torch.stack([
                    torch.cos(base_rpy[:, i, 2]),
                    torch.sin(base_rpy[:, i, 2])
                ], dim=1)
                toward_reward = torch.sum(ideal_toward * self_toward, dim=1) * self.toward_reward_scale
                reward_logger.append(torch.sum(toward_reward).cpu())
                reward[:, i] += toward_reward.squeeze(-1)

        # calculate push reward for each agent
        if self.push_reward_scale != 0:
            push_reward = torch.zeros((self.env.num_envs,), device=self.env.device)
            push_reward[torch.norm(self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 7:9],
                                   dim=1) > 0.1] = self.push_reward_scale
            reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)

        # calculate collision punishment
        if self.collision_punishment_scale != 0:
            punishment_logger = []
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    distance = torch.norm(base_pos[:, i, :] - base_pos[:, j, :], dim=1, keepdim=True)
                    collsion_punishment = (1 / (0.02 + distance / 3)) * self.collision_punishment_scale
                    punishment_logger.append(torch.sum(collsion_punishment).cpu())
                    reward[:, i] += collsion_punishment.squeeze(-1)
                    reward[:, j] += collsion_punishment.squeeze(-1)

        # calculate exception punishment
        if self.exception_punishment_scale != 0:
            reward[self.exception_buf, :] += self.exception_punishment_scale
            reward[self.value_exception_buf, :] += self.exception_punishment_scale
            # reward[self.time_out_buf, :] += self.exception_punishment_scale

        # calculate OCB reward for each agent
        if self.ocb_reward_scale != 0:
            if getattr(self.cfg.rewards, "expanded_ocb_reward", False):
                original_target_direction = (target_pos[:, :2] - box_pos[:, :2]) / (
                    torch.norm((target_pos[:, :2] - box_pos[:, :2] + 0.01), dim=1, keepdim=True))
                delta_yaw = target_rpy[:, 2] - box_rpy[:, 2]
                # delta_yaw -->(-pi, pi)
                delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                # rotate target direction by delta_yaw/2
                target_direction = torch.stack([original_target_direction[:, 0] * torch.cos(
                    -delta_yaw / 2) - original_target_direction[:, 1] * torch.sin(-delta_yaw / 2),
                                                original_target_direction[:, 0] * torch.sin(
                                                    -delta_yaw / 2) + original_target_direction[:, 1] * torch.cos(
                                                    -delta_yaw / 2)], dim=1)
                pass
            else:
                target_direction = (target_pos[:, :2] - box_pos[:, :2]) / (
                    torch.norm((target_pos[:, :2] - box_pos[:, :2]), dim=1, keepdim=True))
            vertex_list = self.cfg.asset.vertex_list
            reward_logger = []
            for i in range(self.num_agents):
                gf_pos = base_pos[:, i, :2] - box_pos[:, :2]
                rotation_matrix = rotation_matrix_2D(- box_rpy[:, 2])
                box_relative_pos = torch.bmm(rotation_matrix, gf_pos.unsqueeze(2)).squeeze(2)
                normal_vector = self.calc_normal_vector_for_obc_reward(vertex_list, box_relative_pos)
                rotation_matrix = rotation_matrix_2D(box_rpy[:, 2])
                normal_vector = torch.bmm(rotation_matrix,
                                          normal_vector.to(rotation_matrix.device).unsqueeze(2)).squeeze(2)
                ocb_reward = torch.sum(target_direction * normal_vector, dim=1) * self.ocb_reward_scale
                reward[:, i] += ocb_reward
                reward_logger.append(torch.sum(ocb_reward).cpu())

        self.last_box_state = deepcopy(box_state)
        self.last_box_pos = deepcopy(box_pos)

        reward = torch.mean(reward, dim=1, keepdim=True).repeat(1, self.num_agents)

        return reward, return_info

