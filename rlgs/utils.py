import torch
import torch.nn as nn


def gradient_clip(model: nn.Module, max_norm: float = 2.4):
    """Apply gradient clipping to model parameters"""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def apply_lr_deltas(optimizer: torch.optim.Optimizer, deltas: torch.Tensor, group_mapping: dict):
    """
    Apply learning rate deltas to optimizer parameter groups.

    Args:
        optimizer: PyTorch optimizer
        deltas: LR delta terms [num_groups] or [batch_size, num_groups]
        group_mapping: Mapping from group name to action index
    """
    # Handle batch dimension if present
    if deltas.dim() > 1:
        deltas = deltas.squeeze(0)

    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "")

        if group_name in group_mapping:
            idx = group_mapping[group_name]
            delta = deltas[idx].item()

            # Store the delta for tracking
            param_group["rl_delta"] = delta
            base = param_group.get("base_lr", param_group["lr"])

            # Apply delta: new_lr = base_lr + delta
            new_lr = base + delta

            # Ensure learning rate stays positive
            param_group["lr"] = max(new_lr, 1e-8)


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
            pg["rl_delta"] = 0.0

            base = pg.get("base_lr", pg["lr"])
            pg["lr"] = base
