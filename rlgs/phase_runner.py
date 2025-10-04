import torch
import random
from typing import Dict, List, Optional, Callable
from .utils import apply_lr_hybrid

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
        policy_lr: float = 1e-3,
        grad_clip: float = 2.4,
    ):
        self.K = K  # Steps per phase
        self.N_lr = N_lr  # Number of LR action trials
        self.policy_lr = policy_lr
        self.grad_clip = grad_clip

    def try_actions(
        self,
        policy,
        state: torch.Tensor,
        gaussians,
        training_views: List,
        render_func: Callable,
        render_args: tuple,
        group_mapping: Dict[str, int],
        original_lrs: Dict[str, float],
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Try multiple actions and return all actions with their rewards.
        Returns a dictionary containing the best action for execution and all actions for policy update.
        """
        best_action = None
        best_reward = float("-inf")
        best_action_idx = None
        new_hidden = hidden_state

        reshuffled_training_views = training_views.copy()
        random.shuffle(reshuffled_training_views)

        # Save initial state
        initial_state = self._save_model_state(gaussians)

        # Sample all actions and evaluate them first
        actions = []  # Will store (global_scale, local_deltas) tuples
        log_probs = []
        sampled_losses = []

        for _ in range(self.N_lr):
            # Sample hybrid action
            global_scale, local_deltas, log_prob, new_hidden = policy.sample_action(state, hidden_state)

            # Evaluate sampled action
            sampled_loss = self._evaluate_action(
                global_scale, local_deltas, gaussians, reshuffled_training_views, render_func, render_args, group_mapping, initial_state
            )

            actions.append((global_scale.detach().clone(), local_deltas.detach().clone()))
            log_probs.append(log_prob)
            sampled_losses.append(sampled_loss)

        # Compute baseline as average of all sampled losses
        baseline_loss = sum(sampled_losses) / len(sampled_losses)

        # Compute rewards for all actions and find the best one
        rewards = []
        for i in range(self.N_lr):
            # Compute reward: R = baseline_loss - M(h)
            reward = baseline_loss - sampled_losses[i]
            rewards.append(reward)

            if reward > best_reward:
                best_reward = reward
                best_action = actions[i]
                best_action_idx = i

        self._restore_model_state(gaussians, initial_state)

        # Return dictionary with all actions and rewards for batch policy update
        return {
            "best_action": best_action,
            "best_action_idx": best_action_idx,
            "all_actions": actions,
            "all_log_probs": log_probs,
            "all_rewards": rewards,
            "best_reward": best_reward,
            "avg_reward": sum(rewards) / len(rewards),
            "hidden_state": new_hidden.detach() if new_hidden is not None else None,
        }

    def _evaluate_action(
        self,
        global_scale: torch.Tensor,
        local_deltas: torch.Tensor,
        gaussians,
        training_views: List,
        render_func: Callable,
        render_args: tuple,
        group_mapping: Dict[str, int],
        initial_state: dict,
    ) -> float:
        """Evaluate a sampled hybrid action and return average loss on all training views"""

        # Restore initial state
        self._restore_model_state(gaussians, initial_state)

        # Apply hybrid learning rate control
        apply_lr_hybrid(gaussians.optimizer, global_scale, local_deltas, group_mapping)

        # Run for number of training views on training views (with gradient updates)
        total_loss = 0.0
        alpha = 0.1  # EMA smoothing factor

        for step in range(len(training_views)):
            viewpoint_cam = training_views[step]
            pose = gaussians.get_RT(viewpoint_cam.uid)

            render_pkg = render_func(viewpoint_cam, gaussians, *render_args, camera_pose=pose)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()

            # Calculate all metrics
            Ll1 = l1_loss(image, gt_image)

            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = 0.8 * Ll1 + 0.2 * (1.0 - ssim_value)

            # Update total_loss using exponential moving average
            if step == 0:
                total_loss = loss.item()
            else:
                total_loss = alpha * loss.item() + (1 - alpha) * total_loss

            loss.backward()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        return total_loss

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
        actions: List[torch.Tensor],
        rewards: List[float],
        log_probs: List[torch.Tensor],
        hidden_state: Optional[torch.Tensor] = None,
    ):
        """Update policy using batch REINFORCE with all sampled actions"""

        # Compute policy loss for all actions
        total_policy_loss = 0.0

        for log_prob, reward in zip(log_probs, rewards):
            # Each action contributes to the loss weighted by its reward
            policy_loss = -log_prob * reward
            total_policy_loss += policy_loss

        # Average the loss over all actions
        avg_policy_loss = total_policy_loss / len(actions)

        # Update policy
        policy_optimizer.zero_grad()
        avg_policy_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)

        policy_optimizer.step()

        return {
            "policy_loss": avg_policy_loss.item(),
            "total_loss": avg_policy_loss.item(),
            "avg_reward": sum(rewards) / len(rewards),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
        }
