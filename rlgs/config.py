from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RLGSConfig:
    """Configuration for RLGS integration"""

    # Enable/disable RLGS
    enabled: bool = True

    # Phase parameters
    K: int = 20  # Steps per phase
    N_lr: int = 3  # Number of LR action trials

    # Policy parameters
    policy_lr: float = 1e-4
    grad_clip: float = 2.4
    entropy_coef: float = 0.01

    # Learning rate groups and bounds
    lr_groups: List[str] = None
    lr_action_bounds: Tuple[float, float] = (0.5, 2.0)

    # Policy architecture
    hidden_dim: int = 64
    state_dim: int = 9  # 3 base + 6 lr_groups (iteration, prev_ssim_loss, prev_l1_loss + rl_scale per group)

    def __post_init__(self):
        if self.lr_groups is None:
            self.lr_groups = ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]

        # Update state_dim to account for base state + rl_scale per group
        self.state_dim = 3 + len(self.lr_groups)


def add_rlgs_args(parser):
    """Add RLGS arguments to argument parser"""
    group = parser.add_argument_group("RLGS Parameters")

    group.add_argument("--rlgs_enabled", action="store_true", default=False, help="Enable RLGS training")
    group.add_argument("--rlgs_K", type=int, default=20, help="Steps per RLGS phase")
    group.add_argument("--rlgs_N_lr", type=int, default=3, help="Number of LR action trials per phase")
    group.add_argument("--rlgs_policy_lr", type=float, default=1e-4, help="Learning rate for RL policy")
    group.add_argument("--rlgs_grad_clip", type=float, default=2.4, help="Gradient clipping for policy")
    group.add_argument("--rlgs_entropy_coef", type=float, default=0.01, help="Entropy coefficient for policy")

    return group


def create_rlgs_config_from_args(args) -> RLGSConfig:
    """Create RLGS config from parsed arguments"""
    return RLGSConfig(
        enabled=getattr(args, "rlgs_enabled", False),
        K=getattr(args, "rlgs_K", 20),
        N_lr=getattr(args, "rlgs_N_lr", 3),
        policy_lr=getattr(args, "rlgs_policy_lr", 1e-4),
        grad_clip=getattr(args, "rlgs_grad_clip", 2.4),
        entropy_coef=getattr(args, "rlgs_entropy_coef", 0.01),
    )
