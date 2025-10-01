import torch
import torch.nn as nn
from typing import Tuple, Optional


class RLLRPolicy(nn.Module):
    """
    GRU-based policy for learning rate scaling (RLLR).
    Maps state to Gaussian action distribution over LR multipliers.
    """

    def __init__(
        self,
        state_dim: int = 9,  # [iteration, prev_ssim_loss, prev_l1_loss] + rl_scale per group
        hidden_dim: int = 64,
        num_groups: int = 6,  # xyz, f_dc, f_rest, opacity, scaling, rotation
        action_bounds: Tuple[float, float] = (0.5, 2.0),
    ):
        super().__init__()

        self.state_dim = state_dim  # [iteration, prev_ssim_loss, prev_l1_loss] + rl_scale per group
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.action_bounds = action_bounds

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

        # Initialize heads to output 1.0 (no scaling initially)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.constant_(self.global_head.bias, 0.0)

    def forward(
        self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy.

        Args:
            state: [batch_size, seq_len, state_dim] or [batch_size, state_dim]
            hidden: Previous hidden state

        Returns:
            local_action_dist: Normal distribution over local actions
            global_action: Deterministic global action [batch_size, 1]
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

    def sample_action(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and return log probability.

        Returns:
            action: Combined action (global * local) [batch_size, num_groups]
            log_prob: Log probability of local action
            new_hidden: Updated hidden state
        """
        local_action_dist, global_raw, new_hidden = self.forward(state, hidden)

        # Sample local actions
        raw_local = local_action_dist.sample()

        # Bound both global and local actions to action_bounds using tanh
        min_val, max_val = self.action_bounds

        # Apply tanh transformation to global action
        tanh_global = torch.tanh(global_raw)
        global_action = tanh_global * (max_val - min_val) / 2 + (max_val + min_val) / 2

        # Apply tanh transformation to local actions
        tanh_local = torch.tanh(raw_local)
        local_action = tanh_local * (max_val - min_val) / 2 + (max_val + min_val) / 2

        # Combine multiplicatively: final_action = global * local
        action = global_action * local_action

        # Log probability is only for the stochastic local actions
        log_prob = local_action_dist.log_prob(raw_local).sum(dim=-1)

        # Jacobian correction for tanh transformation of local actions
        jacobian = (1 - tanh_local**2) * (max_val - min_val) / 2
        jacobian_correction = torch.log(jacobian)
        log_prob += jacobian_correction.sum(dim=-1)

        return action, log_prob, new_hidden
