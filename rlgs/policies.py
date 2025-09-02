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
        state_dim: int = 2,
        hidden_dim: int = 64,
        num_groups: int = 5,
        action_bounds: Tuple[float, float] = (0.5, 2.0),
    ):
        super().__init__()

        self.state_dim = state_dim  # [prev_loss, iteration]
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups  # pos, scale, rot, opacity, sh_base
        self.action_bounds = action_bounds

        # GRU for sequential state processing
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)

        # Output heads for mean and log_std
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

        # Initialize mean head to output 1.0 (no scaling initially)
        nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        """
        Forward pass through policy.

        Args:
            state: [batch_size, seq_len, state_dim] or [batch_size, state_dim]
            hidden: Previous hidden state

        Returns:
            action_dist: Normal distribution over actions
            new_hidden: Updated hidden state
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension

        # GRU forward pass
        gru_out, new_hidden = self.gru(state, hidden)
        gru_out = gru_out[:, -1]  # Take last timestep

        # Compute mean and std
        mean = self.mean_head(gru_out)
        log_std = self.log_std_head(gru_out)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-2.0, max=2.0)
        std = torch.exp(log_std)

        # Create Normal distribution
        action_dist = torch.distributions.Normal(mean, std)

        return action_dist, new_hidden

    def sample_action(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and return log probability.

        Returns:
            action: Sampled action [batch_size, num_groups]
            log_prob: Log probability of action
            new_hidden: Updated hidden state
        """
        action_dist, new_hidden = self.forward(state, hidden)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=-1)

        # Apply bounds using sigmoid
        min_val, max_val = self.action_bounds
        action = torch.sigmoid(action) * (max_val - min_val) + min_val

        return action, log_prob, new_hidden

    def compute_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log probability of given action"""
        # Transform action back to unbounded space
        min_val, max_val = self.action_bounds

        normalized_action = (action - min_val) / (max_val - min_val)
        unbounded_action = torch.logit(torch.clamp(normalized_action, 1e-6, 1 - 1e-6))

        action_dist, _ = self.forward(state, hidden)
        log_prob = action_dist.log_prob(unbounded_action).sum(dim=-1)

        return log_prob
