import torch
import torch.nn as nn
from typing import Optional


class StateEncoder:
    """
    Encodes training state for RL policies.
    State includes previous phase loss and iteration information.
    """

    def __init__(self, max_iterations: int = 30000):
        """
        Args:
            max_iterations: Maximum training iterations for normalization
        """
        self.max_iterations = max_iterations
        self.prev_phase_loss = None

    def encode_state(self, prev_phase_loss: Optional[float], iteration: int) -> torch.Tensor:
        """
        Encode current training state.

        Args:
            prev_phase_loss: Average loss from previous phase (None for first phase)
            iteration: Current iteration number

        Returns:
            state: [loss_normalized, iteration_normalized] tensor
        """
        # Handle first phase
        if prev_phase_loss is None:
            loss_normalized = 0.5  # Neutral starting value for first phase
        else:
            # Use loss directly since it's already in [0,1] range
            loss_normalized = prev_phase_loss

        # Normalize iteration progress
        iteration_normalized = min(iteration / self.max_iterations, 1.0)

        state = torch.tensor([loss_normalized, iteration_normalized], dtype=torch.float32, device="cuda")

        return state.unsqueeze(0)  # Add batch dimension

    def update_phase_loss(self, phase_loss: float):
        """Update the stored phase loss for next state encoding"""
        self.prev_phase_loss = phase_loss

    def get_prev_phase_loss(self) -> Optional[float]:
        """Get the previous phase loss"""
        return self.prev_phase_loss
