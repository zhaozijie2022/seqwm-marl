
from seqwm.envs.quadruped.mqe import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from seqwm.envs.quadruped.mqe import LEGGED_GYM_ROOT_DIR

from seqwm.envs.quadruped.mqe.envs.go1.go1 import Go1


class Go1Object(Go1):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):

        self.npc_collision = getattr(cfg.asset, "npc_collision", True)
        self.fix_npc_base_link = getattr(cfg.asset, "fix_npc_base_link", False)
        self.npc_gravity = getattr(cfg.asset, "npc_gravity", True)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _step_npc(self):
        return

    def _prepare_npc(self):

        self.init_state_npc = getattr(self.cfg.init_state, "init_states_npc")
        if getattr(self.cfg.env, "num_obs", 0) != 0:
            self.obs_states = getattr(self.cfg.obstacle_state, "states_obs")
            self.init_state_npc = self.init_state_npc + self.obs_states
        if hasattr(self.cfg.init_state, "default_npc_joint_angles"):
            self.default_dof_pos_npc = torch.tensor(self.cfg.init_state.default_npc_joint_angles, dtype=torch.float,
                                                    device=self.device, requires_grad=False).reshape(1, -1)
        else:
            self.default_dof_pos_npc = torch.zeros(self.num_actions_npc, dtype=torch.float, device=self.device,
                                                   requires_grad=False).unsqueeze(0)

        # creat npc asset
        asset_path_npc = self.cfg.asset.file_npc.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root_npc = os.path.dirname(asset_path_npc)
        asset_file_npc = os.path.basename(asset_path_npc)
        asset_options_npc = gymapi.AssetOptions()
        asset_options_npc.fix_base_link = self.fix_npc_base_link
        asset_options_npc.disable_gravity = not self.npc_gravity
        self.asset_npc = self.gym.load_asset(self.sim, asset_root_npc, asset_file_npc, asset_options_npc)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.asset_npc)
        friction_range = getattr(self.cfg.domain_rand, "friction_range", [0.5, 0.5])
        for rigid_shape_prop in rigid_shape_props_asset:
            rigid_shape_prop.friction = friction_range[0] + torch.rand(1).item() * (
                        friction_range[1] - friction_range[0])
        self.gym.set_asset_rigid_shape_properties(self.asset_npc, rigid_shape_props_asset)

        # creat target asset
        _asset_path_npc = self.cfg.asset._file_npc.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        _asset_root_npc = os.path.dirname(_asset_path_npc)
        _asset_file_npc = os.path.basename(_asset_path_npc)
        _asset_options_npc = gymapi.AssetOptions()
        _asset_options_npc.fix_base_link = self.cfg.asset._fix_npc_base_link
        _asset_options_npc.disable_gravity = not self.cfg.asset._npc_gravity
        self._asset_npc = self.gym.load_asset(self.sim, _asset_root_npc, _asset_file_npc, _asset_options_npc)

        # creat final target
        if getattr(self.cfg.asset, "file_npc_final", None) != None:
            _asset_path_npc1 = self.cfg.asset.file_npc_final.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            _asset_root_npc1 = os.path.dirname(_asset_path_npc1)
            _asset_file_npc1 = os.path.basename(_asset_path_npc1)
            _asset_options_npc1 = gymapi.AssetOptions()
            _asset_options_npc1.fix_base_link = self.cfg.asset.fix_npc_base_link_final
            _asset_options_npc1.disable_gravity = not self.cfg.asset.npc_gravity_final
            self._asset_npc1 = self.gym.load_asset(self.sim, _asset_root_npc1, _asset_file_npc1, _asset_options_npc1)

        # create obstacle asset
        if getattr(self.cfg.env, "num_obs", 0) != 0:
            obs_asset_path_npc = self.cfg.asset.obs_file_npc.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            obs_asset_root_npc = os.path.dirname(obs_asset_path_npc)
            obs_asset_file_npc = os.path.basename(obs_asset_path_npc)
            obs_asset_options_npc = gymapi.AssetOptions()
            obs_asset_options_npc.fix_base_link = self.cfg.asset.obs_fix_npc_base_link
            obs_asset_options_npc.disable_gravity = not self.cfg.asset.obs_npc_gravity
            self.obs_asset_npc = self.gym.load_asset(self.sim, obs_asset_root_npc, obs_asset_file_npc,
                                                     obs_asset_options_npc)

        # init npc state
        init_state_list_npc = []
        self.start_pose_npc = gymapi.Transform()
        for idx, init_state_npc in enumerate(self.init_state_npc):
            base_init_state_list_npc = init_state_npc.pos + init_state_npc.rot + init_state_npc.lin_vel + init_state_npc.ang_vel
            base_init_state_npc = to_torch(base_init_state_list_npc, device=self.device, requires_grad=False)
            init_state_list_npc.append(base_init_state_npc)
            if idx == 0:
                self.start_pose_npc.p = gymapi.Vec3(*base_init_state_npc[:3])
        self.base_init_state_npc = torch.stack(init_state_list_npc, dim=0).repeat(self.num_envs, 1)
        if getattr(self.cfg.goal, "sequential_goal_pos", False):
            init_target_pos = self.cfg.goal.goal_poses[0]
            self.base_init_state_npc[1, :7] = torch.tensor(init_target_pos, dtype=torch.float, device=self.device,
                                                           requires_grad=False)

    def _create_npc(self, env_handle, env_id):

        npc_handles = []
        # create physical box
        pos = self.env_origins[env_id].clone()
        self.start_pose_npc.p = gymapi.Vec3(*pos)
        npc_handle = self.gym.create_actor(env_handle, self.asset_npc, self.start_pose_npc, self.cfg.asset.name_npc,
                                           env_id, not self.npc_collision, 0)
        rigid_body_props = self.gym.get_actor_rigid_shape_properties(env_handle, npc_handle)
        for rigid_body_prop in rigid_body_props:
            rigid_body_prop.friction = 0.5
        # friction_range = getattr(self.cfg.domain_rand, "friction_range", [0.5, 0.5])
        # for rigid_body_prop in rigid_body_props:
        #     rigid_body_prop.friction = friction_range[0] + torch.rand(1).item() * (
        #                 friction_range[1] - friction_range[0])
        self.gym.set_actor_rigid_shape_properties(env_handle, npc_handle, rigid_body_props)
        npc_handles.append(npc_handle)
        # create target box illusion
        npc_handle = self.gym.create_actor(env_handle, self._asset_npc, self.start_pose_npc, self.cfg.asset._name_npc,
                                           env_id, not self.cfg.asset._npc_collision, 0)
        npc_handles.append(npc_handle)
        # create final target illusion
        if getattr(self.cfg.asset, "file_npc_final", None) != None:
            npc_handle = self.gym.create_actor(env_handle, self._asset_npc1, self.start_pose_npc,
                                               self.cfg.asset.name_npc_final, env_id,
                                               not self.cfg.asset.npc_collision_final, 0)
            npc_handles.append(npc_handle)
        # create obstacle
        num_obs = getattr(self.cfg.env, "num_obs", 0)
        for _ in range(num_obs):
            npc_handle = self.gym.create_actor(env_handle, self.obs_asset_npc, self.start_pose_npc,
                                               self.cfg.asset.obs_name_npc, env_id,
                                               not self.cfg.asset.obs_npc_collision, 0)
            npc_handles.append(npc_handle)
        return npc_handles