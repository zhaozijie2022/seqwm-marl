"""SeqWM actor."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from seqwm.utils.envs_tools import get_shape_from_obs_space
from seqwm.utils.envs_tools import check
from seqwm.utils.models_tools import update_linear_schedule
from seqwm.models.base.wm_networks import create_mlp
from typing import Dict

LOG_STD_MAX = 2
LOG_STD_MIN = -10


class SeqWMPolicy(nn.Module):
    def __init__(self, args: Dict, obs_space, action_space, device=torch.device("cpu")):
        super(SeqWMPolicy, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.log_std_min = torch.tensor(args.get("log_std_min", LOG_STD_MIN)).to(**self.tpdv)
        self.log_std_max = torch.tensor(args.get("log_std_max", LOG_STD_MAX)).to(**self.tpdv)
        act_dim = action_space.shape[0]

        self.net = create_mlp(
            in_dim=get_shape_from_obs_space(obs_space)[0],
            mlp_dims=args["hidden_sizes"][:-1],
            out_dim=args["hidden_sizes"][-1],
            act=None,
            device=device,
        )
        self.mu_layer = nn.Linear(args["hidden_sizes"][-1], act_dim).to(device)
        self.log_std_layer = nn.Linear(args["hidden_sizes"][-1], act_dim).to(device)
        self.act_limit = torch.tensor(action_space.high[0]).to(**self.tpdv)  # the same action dim
        self.to(device)

    def forward(self, obs, stochastic=True, with_logprob=False):
        # Return output from network scaled to action space limits.
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = self._log_std(log_std, self.log_std_min, self.log_std_max)

        if stochastic:
            eps = torch.randn_like(mu).to(**self.tpdv)
            pi = mu + eps * log_std.exp()
        else:
            eps = torch.zeros_like(mu).to(**self.tpdv)
            pi = mu

        if with_logprob:
            log_pi = self.gaussian_logprob(eps, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = self.squash(mu, pi, log_pi)

        return pi, log_pi

    def gaussian_logprob(self, eps, log_std, size=None):
        """Compute Gaussian log probability."""
        residual = self._gaussian_residual(eps, log_std).sum(-1, keepdim=True)
        if size is None:
            size = eps.size(-1)
        return self._gaussian_logprob(residual) * size

    def squash(self, mu, pi, log_pi):
        """Apply squashing function."""
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)

        if log_pi is not None:
            log_pi -= self._squash(pi).sum(-1, keepdim=True)

        mu = mu * self.act_limit
        pi = pi * self.act_limit
        return mu, pi, log_pi

    # region static math methods
    @staticmethod
    @torch.jit.script
    def _gaussian_residual(eps, log_std):
        return -0.5 * eps.pow(2) - log_std

    @staticmethod
    @torch.jit.script
    def _gaussian_logprob(residual):
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=residual.dtype, device=residual.device))
        return residual - 0.5 * log_two_pi

    @staticmethod
    @torch.jit.script
    def _squash(pi):
        return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)

    @staticmethod
    @torch.jit.script
    def _log_std(x, low, high):
        return low + 0.5 * (high - low) * (torch.tanh(x) + 1)
    # endregion


class SeqWM:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert act_space.__class__.__name__ == "Box", "SeqWM only supports continuous action space"

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = args["lr"]
        self.device = device

        self.actor = SeqWMPolicy(args, obs_space, act_space, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.turn_off_grad()

    def get_actions(self, obs, available_actions=None, stochastic=True):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        assert available_actions is None, "SeqWM only supports continuous action space"

        obs = check(obs).to(**self.tpdv)
        actions, _ = self.actor(obs, stochastic=stochastic, with_logprob=False)
        return actions

    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True):
        """Get actions and logprobs of actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (batch_size, dim)
            logp_actions: (torch.Tensor) log probabilities of actions taken by this actor, shape is (batch_size, 1)
        """
        assert available_actions is None, "SeqWM only supports continuous action space"

        obs = check(obs).to(**self.tpdv)
        actions, logp_actions = self.actor(
            obs, stochastic=stochastic, with_logprob=True
        )
        return actions, logp_actions

    def lr_decay(self, step, steps):
        update_linear_schedule(self.actor_optimizer, step, steps, self.lr)


    def save(self, save_dir, agent_id):
        """Save the actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(agent_id) + ".pt"
        )

    def restore(self, model_dir, agent_id):
        """Restore the actor."""
        actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(agent_id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)

    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = False


