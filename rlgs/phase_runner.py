import torch
import copy
from typing import Dict, List, Tuple, Optional, Callable
from .utils import apply_lr_scaling, restore_optimizer_lrs

from utils.loss_utils import l1_loss, ssim

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False


class PhaseRunner:
    """
    Manages phase-based training with RL policy trials and rollouts.
    Handles action sampling, evaluation, and policy updates.
    """

    def __init__(
        self,
        K: int = 20,
        N_lr: int = 3,
        policy_lr: float = 1e-4,
        grad_clip: float = 2.4,
        entropy_coef: float = 0.01,
    ):
        self.K = K  # Steps per phase
        self.N_lr = N_lr  # Number of LR action trials
        self.policy_lr = policy_lr
        self.grad_clip = grad_clip
        self.entropy_coef = entropy_coef

    def try_actions(
        self,
        policy,
        state: torch.Tensor,
        gaussians,
        reward_views: List,
        render_func: Callable,
        render_args: tuple,
        group_mapping: Dict[str, int],
        original_lrs: Dict[str, float],
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        Try multiple actions and return the best one based on reward.
        """
        best_action = None
        best_log_prob = None
        best_reward = float("-inf")
        new_hidden = hidden_state

        # Save initial state
        initial_state = self._save_model_state(gaussians)

        action = torch.ones(len(group_mapping))

        baseline_loss = self._evaluate_action(action, gaussians, reward_views, render_func, render_args, group_mapping, initial_state)

        for _ in range(self.N_lr):
            # Sample action
            action, log_prob, new_hidden = policy.sample_action(state, hidden_state)

            # Evaluate sampled action
            sampled_loss = self._evaluate_action(action, gaussians, reward_views, render_func, render_args, group_mapping, initial_state)

            # Compute reward: R = M(h_orig) - M(h)
            reward = baseline_loss - sampled_loss

            if reward > best_reward:
                best_reward = reward
                best_action = action.detach().clone()
                best_log_prob = log_prob.detach().clone()

        self._restore_model_state(gaussians, initial_state)

        return best_action, best_log_prob, best_reward, new_hidden.detach() if new_hidden is not None else None

    def _evaluate_action(
        self,
        action: torch.Tensor,
        gaussians,
        reward_views: List,
        render_func: Callable,
        render_args: tuple,
        group_mapping: Dict[str, int],
        initial_state: dict,
    ) -> float:
        """Evaluate a sampled action and return average loss"""

        # Restore initial state
        self._restore_model_state(gaussians, initial_state)

        # Apply learning rate scaling
        apply_lr_scaling(gaussians.optimizer, action, group_mapping)

        total_loss = 0.0

        for step in range(self.K):
            view_idx = step % len(reward_views)
            viewpoint_cam = reward_views[view_idx]
            pose = gaussians.get_RT(viewpoint_cam.uid)

            render_pkg = render_func(viewpoint_cam, gaussians, *render_args, camera_pose=pose)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)

            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim_value)

            loss.backward()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

        # Restore original learning rates
        restore_optimizer_lrs(gaussians.optimizer, group_mapping)

        return total_loss / self.K

    def _save_model_state(self, gaussians) -> dict:
        ms = {
            "_xyz": gaussians._xyz.detach().cpu(),
            "_features_dc": gaussians._features_dc.detach().cpu(),
            "_features_rest": gaussians._features_rest.detach().cpu(),
            "_scaling": gaussians._scaling.detach().cpu(),
            "_rotation": gaussians._rotation.detach().cpu(),
            "_opacity": gaussians._opacity.detach().cpu(),
            "P": gaussians.P.detach().cpu() if hasattr(gaussians, "P") else None,
        }

        opt_state = gaussians.optimizer.state_dict()

        for st in opt_state["state"].values():
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor):
                    st[k] = v.detach().cpu()

        return {"model_state": ms, "optimizer_state": opt_state}

    def _restore_model_state(self, gaussians, saved_state: dict):
        ms = saved_state["model_state"]
        gaussians._xyz.data.copy_(ms["_xyz"].to("cuda"))
        gaussians._features_dc.data.copy_(ms["_features_dc"].to("cuda"))
        gaussians._features_rest.data.copy_(ms["_features_rest"].to("cuda"))
        gaussians._scaling.data.copy_(ms["_scaling"].to("cuda"))
        gaussians._rotation.data.copy_(ms["_rotation"].to("cuda"))
        gaussians._opacity.data.copy_(ms["_opacity"].to("cuda"))

        if ms["P"] is not None:
            gaussians.P.data.copy_(ms["P"].to("cuda"))

        opt_state = saved_state["optimizer_state"]

        for st in opt_state["state"].values():
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor):
                    st[k] = v.to("cuda")

        gaussians.optimizer.load_state_dict(opt_state)

    def update_policy(
        self,
        policy,
        policy_optimizer: torch.optim.Optimizer,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        log_prob: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        """Update policy using vanilla REINFORCE with entropy bonus"""

        # Policy loss: -log_prob * reward (no baseline subtraction)
        policy_loss = -log_prob * reward

        # Entropy bonus
        action_dist, _ = policy.forward(state, hidden_state)
        entropy = action_dist.entropy().sum()
        entropy_loss = -self.entropy_coef * entropy

        # Total loss
        total_loss = policy_loss + entropy_loss

        # Update policy
        policy_optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)

        policy_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "reward": reward,
        }
