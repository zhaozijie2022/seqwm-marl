import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class SimNorm(nn.Module):
    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
        Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"bias={self.bias is not None}{repr_dropout}, " \
               f"act={self.act.__class__.__name__})"


class ActedLinear(nn.Linear):
    """
        Linear layer with activation
    """

    def __init__(self, *args, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.act = act

    def forward(self, x):
        x = super().forward(x)
        return self.act(x)

    def __repr__(self):
        return f"ActedLinear(in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"bias={self.bias is not None}, " \
               f"act={self.act.__class__.__name__})"


def create_mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0., device='cpu', normed=True):
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    if normed:
        for i in range(len(dims) - 2):
            mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
        mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    else:
        for i in range(len(dims) - 2):
            mlp.append(ActedLinear(dims[i], dims[i + 1], act=act, dropout=dropout * (i == 0)))
        mlp.append(nn.Linear(dims[-2], dims[-1]))

    return nn.Sequential(*mlp).to(device)


class MLP(nn.Module):
    def __init__(self, in_dim, mlp_dims, out_dim, act=None, dropout=0., lr=1e-3, device='cpu'):
        super().__init__()
        self.mlp = create_mlp(in_dim, mlp_dims, out_dim, act, dropout, device)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr
        )
        self.to(device)

    def forward(self, x):
        return self.mlp(x)

    def turn_on_grad(self):
        for param in self.mlp.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        for param in self.mlp.parameters():
            param.requires_grad = False

    def save(self, **kwargs):
        assert "save_dir" in kwargs
        assert "model_name" in kwargs
        if "agent_id" in kwargs:
            torch.save(
                self.mlp.state_dict(), str(kwargs["save_dir"]) + '/' + kwargs["model_name"] + str(kwargs["agent_id"]) + ".pt"
            )
        else:
            torch.save(
                self.mlp.state_dict(), str(kwargs["save_dir"]) + '/' + kwargs["model_name"] + ".pt"
            )

    def restore(self, **kwargs):
        assert "load_dir" in kwargs
        assert "model_name" in kwargs
        if "agent_id" in kwargs:
            self.mlp.load_state_dict(
                torch.load(str(kwargs["load_dir"]) + '/' + kwargs["model_name"] + str(kwargs["agent_id"]) + ".pt")
            )
        else:
            self.mlp.load_state_dict(
                torch.load(str(kwargs["load_dir"]) + '/' + kwargs["model_name"] + ".pt")
            )


class MLPEncoder(MLP):
    def __init__(self, in_dim, mlp_dims, out_dim, act=None, dropout=0., lr=1e-3, device='cpu'):
        super().__init__(in_dim, mlp_dims, out_dim, act, dropout, lr, device)

    def encode(self, x):
        """
        Args:
            x:  (batch_size, dim)
        Returns:
            latent: (batch_size, latent_dim)
        """
        return self.forward(x)


class MLPPredictor(MLP):
    def __init__(self, in_dim, mlp_dims, out_dim, act=None, dropout=0., lr=1e-3, device='cpu'):
        super().__init__(in_dim, mlp_dims, out_dim, act, dropout, lr, device)

    def predict(self, z, a):
        """
        Args:
            z: (batch_size, latent_dim * num_agents)
            a: (batch_size, action_dim_sum)
        Returns:
            next_state_latent_pred: (batch_size, latent_dim)
            reward_pred_logits: (batch_size, num_bins)
        """
        x = torch.cat([z, a], dim=-1)
        return self.forward(x)


class OneHotProcessor:
    def __init__(self, num_bins, vmin, vmax, device):
        self.num_bins = num_bins
        self.vmin, self.vmax = vmin, vmax
        if num_bins > 1:
            self.bin_size = (vmax - vmin) / (num_bins - 1)
            self.dis_reg_bins = torch.linspace(vmin, vmax, num_bins, device=device)
        else:
            self.bin_size = 0.0
            self.dis_reg_bins = None
        self.sym_log = lambda x: torch.sign(x) * torch.log(1 + torch.abs(x))
        self.sym_exp = lambda x: torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def logits_decode_scalar(self, logits):
        if self.num_bins == 0:
            return logits
        elif self.num_bins == 1:
            return self.sym_exp(logits)
        else:
            bin_indices = torch.argmax(logits, dim=-1, keepdim=True)
            return self.dis_reg_bins[bin_indices]

    def scalar_encode_logits(self, x):
        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return self.sym_log(x)
        else:
            bin_size = (self.vmax - self.vmin) / (self.num_bins - 1)
            x_clamped = torch.clamp(x, self.vmin, self.vmax).squeeze(-1)
            bin_idx = torch.round((x_clamped - self.vmin) / bin_size)
            bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1).long()
            return F.one_hot(bin_idx, self.num_bins).float()

    def dis_reg_loss(self, logits, target):
        if self.num_bins <= 1:
            return F.mse_loss(self.logits_decode_scalar(logits), target)
        else:
            target = self.scalar_encode_logits(target)
            return F.cross_entropy(logits, target)

    def reg_acc(self, logits, target):
        """
        Args:
            logits: (batch_size, num_bins)
            target: (batch_size, 1)
        """
        target = self.scalar_encode_logits(target)
        bin_indices = torch.argmax(logits, dim=-1)
        return torch.sum(bin_indices == target.squeeze(-1)).float() / target.size(0)



class TwoHotProcessor:
    def __init__(self, num_bins, vmin, vmax, device):
        self.num_bins = num_bins
        self.vmin, self.vmax = vmin, vmax
        if num_bins > 1:
            self.bin_size = (vmax - vmin) / (num_bins - 1)
            self.dis_reg_bins = torch.linspace(vmin, vmax, num_bins, device=device)
        else:
            self.bin_size = 0.0
            self.dis_reg_bins = None

        self.sym_log = lambda x: torch.sign(x) * torch.log(1 + torch.abs(x))
        self.sym_exp = lambda x: torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def logits_decode_scalar(self, x):
        """logits -> scalars：[*, num_bins] → [*, 1]）"""
        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return self.sym_exp(x)
        else:
            x_softmax = F.softmax(x, dim=-1)
            weighted_sum = torch.sum(x_softmax * self.dis_reg_bins, dim=-1, keepdim=True)
            return self.sym_exp(weighted_sum)

    def scalar_encode_logits(self, x):
        """scalars -> two-hot [batch_size, 1] -> [batch_size, num_bins]"""
        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return self.sym_log(x)
        else:
            x_sym_log = self.sym_log(x)
            x_clamped = torch.clamp(x_sym_log, self.vmin, self.vmax).squeeze(1)

            bin_idx = torch.floor((x_clamped - self.vmin) / self.bin_size).long()
            bin_offset = ((x_clamped - self.vmin) / self.bin_size - bin_idx.float()).unsqueeze(-1)

            soft_two_hot = torch.zeros(x.size(0), self.num_bins, device=x.device)
            next_bin = (bin_idx + 1) % self.num_bins

            soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
            soft_two_hot.scatter_(1, next_bin.unsqueeze(1), bin_offset)
            return soft_two_hot

    def dis_reg_loss(self, logits, target):
        if self.num_bins == 0:
            return F.mse_loss(logits, target)
        elif self.num_bins == 1:
            return F.mse_loss(self.logits_decode_scalar(logits), target)
        else:
            log_pred = F.log_softmax(logits, dim=-1)
            target = self.scalar_encode_logits(target)
            return -(target * log_pred).sum(dim=-1, keepdim=True)


class RunningScale:
    def __init__(self, tpdv, tau):
        self._value = torch.ones(1).to(**tpdv)
        self._percentiles = torch.tensor([5, 95]).to(**tpdv)
        self.tau = tau

    @property
    def value(self):
        return self._value.cpu().item()

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x, dim=0)
        positions = self._percentiles * (x.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        return (d0 + d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.)
        self._value.data.lerp_(value, self.tau)

    def state_dict(self):
        return dict(value=self._value, percentiles=self._percentiles)

    def load_state_dict(self, state_dict):
        self._value.data.copy_(state_dict['value'])
        self._percentiles.data.copy_(state_dict['percentiles'])

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1 / self.value)

    def __repr__(self):
        return f'RunningScale(S: {self.value})'


