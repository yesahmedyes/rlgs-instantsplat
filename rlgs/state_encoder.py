import torch
from typing import Optional, List


class StateEncoder:
    """
    Encodes training state for RL policies.
    State includes previous phase losses and iteration information.
    """

    def __init__(self, max_iterations: int = 30000, lr_groups: Optional[List[str]] = None):
        self.max_iterations = max_iterations
        self.prev_ssim_loss = None
        self.prev_l1_loss = None
        self.lr_groups = lr_groups or ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]

    def encode_state(
        self,
        iteration: int,
        prev_ssim_loss: float = 1.0,
        prev_l1_loss: float = 1.0,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> torch.Tensor:
        """
        Args:
            iteration: Current iteration number
            prev_ssim_loss: SSIM loss from previous phase
            prev_l1_loss: L1 loss from previous phase
            optimizer: If provided, appends per-group rl_scale features
        """
        iteration_normalized = min(iteration / self.max_iterations, 1.0)

        comps = [iteration_normalized, prev_ssim_loss, prev_l1_loss]

        if optimizer is not None:
            # Only use the groups specified in lr_groups, in that order
            for name in self.lr_groups:
                pg = next((g for g in optimizer.param_groups if g.get("name", "") == name), None)

                if pg is None:
                    comps.append(1.0)
                else:
                    scale = pg.get("rl_scale", 1.0)
                    comps.append(float(scale))

        state = torch.tensor(comps, dtype=torch.float32, device="cuda")

        return state.unsqueeze(0)

    def update_losses(self, ssim_loss: float, l1_loss: float):
        self.prev_ssim_loss = ssim_loss
        self.prev_l1_loss = l1_loss
