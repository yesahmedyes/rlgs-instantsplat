import torch
import torch.nn as nn


def gradient_clip(model: nn.Module, max_norm: float = 2.4):
    """Apply gradient clipping to model parameters"""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def apply_lr_hybrid(optimizer: torch.optim.Optimizer, global_scale: torch.Tensor, local_deltas: torch.Tensor, group_mapping: dict):
    """
    Apply hybrid LR control to optimizer parameter groups.
    Final LR = base_lr * global_scale + local_delta

    Args:
        optimizer: PyTorch optimizer
        global_scale: Global scaling factor [1] or [batch_size, 1]
        local_deltas: Local delta terms [num_groups] or [batch_size, num_groups]
        group_mapping: Mapping from group name to action index
    """
    # Handle batch dimension if present
    if global_scale.dim() > 1:
        global_scale = global_scale.squeeze(0)
    if local_deltas.dim() > 1:
        local_deltas = local_deltas.squeeze(0)

    global_scale_val = global_scale.item()

    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "")

        if group_name in group_mapping:
            idx = group_mapping[group_name]
            local_delta = local_deltas[idx].item()

            # Store both components for tracking
            param_group["rl_global_scale"] = global_scale_val
            param_group["rl_local_delta"] = local_delta
            base = param_group.get("base_lr", param_group["lr"])

            # Apply hybrid formula: new_lr = base_lr * global_scale + local_delta
            new_lr = base * global_scale_val + local_delta

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
            pg["rl_global_scale"] = 1.0
            pg["rl_local_delta"] = 0.0

            base = pg.get("base_lr", pg["lr"])
            pg["lr"] = base
