import torch
import torch.nn as nn


def gradient_clip(model: nn.Module, max_norm: float = 2.4):
    """Apply gradient clipping to model parameters"""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def compute_entropy(action_dist) -> torch.Tensor:
    """Compute entropy of action distribution"""
    return action_dist.entropy().sum(dim=-1)


def compute_log_prob(action_dist, action: torch.Tensor) -> torch.Tensor:
    """Compute log probability of action"""
    return action_dist.log_prob(action).sum(dim=-1)


def apply_lr_scaling(optimizer: torch.optim.Optimizer, action: torch.Tensor, group_mapping: dict, original_lrs: dict):
    """
    Apply learning rate scaling to optimizer parameter groups.

    Args:
        optimizer: PyTorch optimizer
        action: LR scaling factors [num_groups]
        group_mapping: Mapping from group name to action index
        original_lrs: Original learning rates for each group
    """
    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "")

        if group_name in group_mapping:
            action_idx = group_mapping[group_name]
            scale_factor = action[action_idx].item()
            original_lr = original_lrs[group_name]
            param_group["lr"] = original_lr * scale_factor


def save_optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict:
    """Save original learning rates from optimizer"""
    original_lrs = {}

    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "")

        if group_name:
            original_lrs[group_name] = param_group["lr"]

    return original_lrs


def restore_optimizer_lrs(optimizer: torch.optim.Optimizer, original_lrs: dict):
    """Restore original learning rates to optimizer"""
    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "")

        if group_name in original_lrs:
            param_group["lr"] = original_lrs[group_name]
