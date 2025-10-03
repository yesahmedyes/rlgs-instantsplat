import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional


class LRRecorder:
    """Records and plots learning rate changes during RLGS training."""

    def __init__(self, lr_groups: List[str], output_dir: str):
        """
        Initialize LR recorder.

        Args:
            lr_groups: List of LR group names (e.g., ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation'])
            output_dir: Directory to save plots and data
        """
        self.lr_groups = lr_groups
        self.output_dir = output_dir

        self.iterations = []
        self.base_lr_data = {group: [] for group in lr_groups}
        self.lr_data = {group: [] for group in lr_groups}
        self.delta_data = {group: [] for group in lr_groups}

        os.makedirs(output_dir, exist_ok=True)

    def record_lrs(self, iteration: int, optimizer: torch.optim.Optimizer, group_mapping: Dict):
        """
        Record current learning rates and delta values.

        Args:
            iteration: Current global step/iteration
            optimizer: The Gaussian optimizer
            group_mapping: Mapping from LR groups to optimizer param groups
        """
        self.iterations.append(iteration)

        for group_name in self.lr_groups:
            if group_name in group_mapping:
                idx = group_mapping[group_name]
                base_lr = optimizer.param_groups[idx]["base_lr"]
                lr = optimizer.param_groups[idx]["lr"]
                delta = optimizer.param_groups[idx].get("rl_delta", 0.0)

                self.lr_data[group_name].append(lr)
                self.base_lr_data[group_name].append(base_lr)
                self.delta_data[group_name].append(delta)

    def plot_lrs(self):
        """
        Create plots showing both base_lr and lr for each group.
        Saves individual plots for each group and a combined overview plot.
        """
        if not self.iterations:
            print("No learning rate data recorded yet.")
            return

        # Create individual plots for each group
        for group_name in self.lr_groups:
            if group_name in self.lr_data and self.lr_data[group_name]:
                plt.figure(figsize=(12, 8))

                # Plot base_lr, current lr, and delta
                plt.subplot(2, 1, 1)
                plt.plot(self.iterations, self.base_lr_data[group_name], label=f"Base LR ({group_name})", linestyle="--", alpha=0.7)
                plt.plot(self.iterations, self.lr_data[group_name], label=f"Current LR ({group_name})", linewidth=2)
                plt.ylabel("Learning Rate")
                plt.title(f"Learning Rate Evolution - {group_name}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale("log")  # Log scale is often better for LR visualization

                # Plot delta values
                plt.subplot(2, 1, 2)
                plt.plot(self.iterations, self.delta_data[group_name], label=f"Delta ({group_name})", color="red", linewidth=2)
                plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
                plt.ylabel("Delta Value")
                plt.xlabel("Iteration")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save individual plot
                plot_path = os.path.join(self.output_dir, f"{group_name}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
