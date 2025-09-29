import torch
import torch.nn as nn


def gradient_clip(model: nn.Module, max_norm: float = 2.4):
    """Apply gradient clipping to model parameters"""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def apply_lr_scaling(optimizer: torch.optim.Optimizer, action: torch.Tensor, group_mapping: dict):
    """
    Apply learning rate scaling to optimizer parameter groups.

    Args:
        optimizer: PyTorch optimizer
        action: LR scaling factors [num_groups] or [batch_size, num_groups]
        group_mapping: Mapping from group name to action index
        original_lrs: Original learning rates for each group
    """
    # Handle batch dimension if present
    if action.dim() > 1:
        action = action.squeeze(0)

    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "")

        if group_name in group_mapping:
            idx = group_mapping[group_name]
            scale = action[idx].item()

            param_group["rl_scale"] = scale
            base = param_group.get("base_lr", param_group["lr"])
            param_group["lr"] = base * scale


def save_optimizer_lrs(optimizer):
    original = {}

    for pg in optimizer.param_groups:
        name = pg.get("name", "")

        if name:
            original[name] = pg.get("base_lr", pg["lr"])

    return original


def restore_optimizer_lrs(optimizer, group_mapping: dict):
    for pg in optimizer.param_groups:
        group_name = pg.get("name", "")

        if group_name in group_mapping:
            pg["rl_scale"] = 1.0

            base = pg.get("base_lr", pg["lr"])
            pg["lr"] = base
