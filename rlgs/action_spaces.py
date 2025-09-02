import torch
from typing import Dict, Tuple, List


class ActionSpaces:
    """
    Defines action spaces and bounds for RLGS policies.
    Handles transformations between bounded and unbounded action spaces.
    """

    def __init__(self, lr_groups: List[str], lr_action_bounds: Tuple[float, float] = (0.5, 2.0)):
        """
        Args:
            lr_groups: List of parameter group names for LR scaling
            lr_action_bounds: (min, max) bounds for LR multipliers
        """
        self.lr_groups = lr_groups
        self.lr_action_bounds = lr_action_bounds
        self.num_lr_groups = len(lr_groups)

    def get_lr_group_mapping(self) -> Dict[str, int]:
        """Get mapping from parameter group name to action index"""
        return {group: i for i, group in enumerate(self.lr_groups)}

    def bounded_to_unbounded(self, bounded_action: torch.Tensor) -> torch.Tensor:
        """Transform bounded action to unbounded space for policy"""
        min_val, max_val = self.lr_action_bounds

        normalized = (bounded_action - min_val) / (max_val - min_val)

        normalized = torch.clamp(normalized, 1e-6, 1 - 1e-6)

        unbounded = torch.logit(normalized)

        return unbounded

    def unbounded_to_bounded(self, unbounded_action: torch.Tensor) -> torch.Tensor:
        """Transform unbounded action to bounded space"""
        min_val, max_val = self.lr_action_bounds

        normalized = torch.sigmoid(unbounded_action)

        bounded = normalized * (max_val - min_val) + min_val

        return bounded

    def get_default_action(self, device: str = "cuda") -> torch.Tensor:
        """Get default action (no scaling, all 1.0)"""
        return torch.ones(self.num_lr_groups, device=device)

    def validate_action(self, action: torch.Tensor) -> torch.Tensor:
        """Ensure action is within valid bounds"""
        min_val, max_val = self.lr_action_bounds

        return torch.clamp(action, min_val, max_val)
