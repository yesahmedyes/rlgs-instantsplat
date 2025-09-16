import torch
from typing import Optional


class StateEncoder:
    """
    Encodes training state for RL policies.
    State includes previous phase losses and iteration information.
    """

    def __init__(self, max_iterations: int = 30000):
        self.max_iterations = max_iterations
        self.prev_ssim_loss = None
        self.prev_l1_loss = None

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
            optimizer: If provided, appends per-group [base_lr, rl_scale] features
            group_order: Optional explicit order of group names
        """
        iteration_normalized = min(iteration / self.max_iterations, 1.0)

        comps = [iteration_normalized, prev_ssim_loss, prev_l1_loss]

        if optimizer is not None:
            group_order = [pg.get("name", "") for pg in optimizer.param_groups if pg.get("name", "")]

            for name in group_order:
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
