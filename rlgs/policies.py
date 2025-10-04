import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class RLLRPolicy(nn.Module):
    """
    GRU-based policy for hybrid LR control (RLLR).
    Maps state to global scaling + local delta terms.
    Final LR = base_lr * global_scale + local_delta
    """

    def __init__(
        self,
        state_dim: int = 9,
        hidden_dim: int = 64,
        num_groups: int = 6,
        # Hybrid bounds: global scaling + local deltas
        global_scale_bounds: Tuple[float, float] = (0.5, 2.0),  # Global scaling
        local_delta_bounds: Tuple[float, float] = (-0.001, 0.001),  # Local deltas
        base_lrs: Optional[Dict[str, float]] = None,  # Reference base LRs for normalization
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.global_scale_bounds = global_scale_bounds
        self.local_delta_bounds = local_delta_bounds
        self.base_lrs = base_lrs or {}

        # GRU for sequential state processing
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)

        # Global action head (deterministic)
        self.global_head = nn.Linear(hidden_dim, 1)

        # Local action heads for mean and log_std (per-group, stochastic)
        self.mean_head = nn.Linear(hidden_dim, num_groups)
        self.log_std_head = nn.Linear(hidden_dim, num_groups)

        # Initialize with small weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable early training"""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        # Initialize heads appropriately
        nn.init.constant_(self.mean_head.bias, 0.0)  # Local deltas start at 0.0
        nn.init.constant_(self.global_head.bias, 0.0)  # Global scale starts at 1.0 (after tanh)

    def forward(
        self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy.

        Args:
            state: [batch_size, seq_len, state_dim] or [batch_size, state_dim]
            hidden: Previous hidden state

        Returns:
            local_action_dist: Normal distribution over local delta actions
            global_action: Deterministic global scale action [batch_size, 1]
            new_hidden: Updated hidden state
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension

        # GRU forward pass
        gru_out, new_hidden = self.gru(state, hidden)
        gru_out = gru_out[:, -1]  # Take last timestep

        # Compute global action (deterministic)
        global_raw = self.global_head(gru_out)

        # Compute local mean and std (stochastic)
        mean = self.mean_head(gru_out)
        log_std = self.log_std_head(gru_out)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-2.0, max=2.0)
        std = torch.exp(log_std)

        # Create Normal distribution for local actions
        local_action_dist = torch.distributions.Normal(mean, std)

        return local_action_dist, global_raw, new_hidden

    def sample_action(
        self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample hybrid actions (global scale + local deltas) and return log probability.

        Returns:
            global_scale: Global scaling factor [batch_size, 1]
            local_deltas: Local delta terms [batch_size, num_groups]
            log_prob: Log probability of local actions
            new_hidden: Updated hidden state
        """
        local_action_dist, global_raw, new_hidden = self.forward(state, hidden)

        # Sample local actions
        raw_local = local_action_dist.sample()

        # Bound global action to scale bounds
        min_scale, max_scale = self.global_scale_bounds
        tanh_global = torch.tanh(global_raw)
        global_scale = tanh_global * (max_scale - min_scale) / 2 + (max_scale + min_scale) / 2

        # Bound local actions to delta bounds
        min_delta, max_delta = self.local_delta_bounds
        tanh_local = torch.tanh(raw_local)
        local_deltas = tanh_local * (max_delta - min_delta) / 2 + (max_delta + min_delta) / 2

        # Log probability is only for the stochastic local actions
        log_prob = local_action_dist.log_prob(raw_local).sum(dim=-1)

        # Jacobian correction for tanh transformation of local actions
        jacobian = (1 - tanh_local**2) * (max_delta - min_delta) / 2
        jacobian_correction = torch.log(jacobian)
        log_prob += jacobian_correction.sum(dim=-1)

        return global_scale, local_deltas, log_prob, new_hidden
