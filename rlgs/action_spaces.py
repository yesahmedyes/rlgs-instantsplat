from typing import Dict, Tuple, List


class ActionSpaces:
    """
    Defines action spaces and bounds for RLGS policies.
    Handles transformations between bounded and unbounded action spaces.
    """

    def __init__(self, lr_groups: List[str], lr_delta_bounds: Tuple[float, float] = (-0.001, 0.001)):
        """
        Args:
            lr_groups: List of parameter group names for LR deltas
            lr_delta_bounds: (min, max) bounds for LR delta terms
        """
        self.lr_groups = lr_groups
        self.lr_delta_bounds = lr_delta_bounds
        self.num_lr_groups = len(lr_groups)

    def get_lr_group_mapping(self) -> Dict[str, int]:
        """Get mapping from parameter group name to action index"""
        return {group: i for i, group in enumerate(self.lr_groups)}
