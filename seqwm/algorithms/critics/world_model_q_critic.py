from copy import deepcopy
import torch
import torch.nn as nn
import itertools
from seqwm.utils.envs_tools import check
from seqwm.utils.envs_tools import get_shape_from_obs_space
from typing import Dict
import seqwm.models.base.wm_networks as wms


def get_combined_dim(cent_obs_feature_dim, act_spaces):
    """Get the combined dimension of central observation and individual actions."""
    combined_dim = cent_obs_feature_dim
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
        else:
            action_dims = space.nvec
            for action_dim in action_dims:
                combined_dim += action_dim
    return combined_dim


class DisRegQNet(nn.Module):
    def __init__(self, args: Dict, cent_obs_space, act_spaces, device=torch.device("cpu")):
        super().__init__()
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)[0]
        num_bins = args.get("num_bins", 101)
        dropout = args.get("dropout", 0.01)
        self.mlp = wms.create_mlp(
            in_dim=get_combined_dim(cent_obs_shape, act_spaces),
            mlp_dims=args["hidden_sizes"],
            out_dim=num_bins,
            dropout=dropout,
        )
        self.to(device)

    def forward(self, cent_obs, actions):
        concat_x = torch.cat([cent_obs, actions], dim=-1)
        q_logits = self.mlp(concat_x)
        return q_logits


class EnsembleDisRegQCritic:
    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
        wm_args=None,
    ):

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = act_space[0].__class__.__name__
        self.critic = DisRegQNet(args, share_obs_space, act_space, device)
        self.critic2 = DisRegQNet(args, share_obs_space, act_space, device)
        self.critic.mlp[-1].weight.data.fill_(0)
        self.critic2.mlp[-1].weight.data.fill_(0)
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        critic_params = itertools.chain(
            self.critic.parameters(), self.critic2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.critic_lr,
        )
        self.processor = wms.TwoHotProcessor(
            num_bins=wm_args["num_bins"],
            vmin=wm_args["reward_min"],
            vmax=wm_args["reward_max"],
            device=device,
        )
        self.scale = wms.RunningScale(
            tpdv=self.tpdv,
            tau=wm_args["scale_tau"]
        )

        self.turn_off_grad()

    def get_values(self, share_obs, actions):
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        q_logits = self.critic(share_obs, actions)
        q_logits2 = self.critic2(share_obs, actions)
        q_value = self.processor.logits_decode_scalar(q_logits)
        q_value2 = self.processor.logits_decode_scalar(q_logits2)
        return torch.min(
            q_value, q_value2
        )

    def get_mean_values(self, share_obs, actions):
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        q_logits = self.critic(share_obs, actions)
        q_logits2 = self.critic2(share_obs, actions)
        q_value = self.processor.logits_decode_scalar(q_logits)
        q_value2 = self.processor.logits_decode_scalar(q_logits2)
        return (q_value + q_value2) / 2

    @torch.no_grad()
    def get_target_values(self, share_obs, actions):
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        q_logits = self.target_critic(share_obs, actions)
        q_logits2 = self.target_critic2(share_obs, actions)
        q_value = self.processor.logits_decode_scalar(q_logits)
        q_value2 = self.processor.logits_decode_scalar(q_logits2)
        return torch.min(
            q_value, q_value2
        )

    def soft_update(self):
        for param_target, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )
        for param_target, param in zip(
                self.target_critic2.parameters(), self.critic2.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def save(self, save_dir):
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent" + ".pt",
        )
        torch.save(self.critic2.state_dict(), str(save_dir) + "/critic_agent2" + ".pt")
        torch.save(
            self.target_critic2.state_dict(),
            str(save_dir) + "/target_critic_agent2" + ".pt",
        )
        torch.save(
            self.scale.state_dict(),
            str(save_dir) + "/q_scale.pt"
        )

    def restore(self, model_dir):
        critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + ".pt")
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + "/target_critic_agent" + ".pt"
        )
        self.target_critic.load_state_dict(target_critic_state_dict)
        critic_state_dict2 = torch.load(str(model_dir) + "/critic_agent2" + ".pt")
        self.critic2.load_state_dict(critic_state_dict2)
        target_critic_state_dict2 = torch.load(
            str(model_dir) + "/target_critic_agent2" + ".pt"
        )
        self.target_critic2.load_state_dict(target_critic_state_dict2)
        scale_state_dict = torch.load(str(model_dir) + "/q_scale.pt")
        self.scale.load_state_dict(scale_state_dict)

    def turn_on_grad(self):
        for param in self.critic.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
































