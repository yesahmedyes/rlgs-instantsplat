import torch
import copy
from typing import Dict, List, Tuple, Optional, Callable
from .utils import apply_lr_scaling, restore_optimizer_lrs


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

        Args:
            policy: RL policy for action sampling
            state: Current training state
            gaussians: Gaussian model
            reward_views: Views for reward computation
            render_func: Rendering function
            render_args: Arguments for rendering
            group_mapping: Parameter group to action mapping
            original_lrs: Original learning rates
            hidden_state: Policy hidden state

        Returns:
            best_action: Best action found
            best_log_prob: Log probability of best action
            best_reward: Best reward achieved
            new_hidden: Updated hidden state
        """
        best_action = None
        best_log_prob = None
        best_reward = float("-inf")
        new_hidden = hidden_state

        # Save initial state
        initial_state = self._save_model_state(gaussians)

        for _ in range(self.N_lr):
            # Sample action
            action, log_prob, new_hidden = policy.sample_action(state, hidden_state)

            # Apply action and run short rollout
            reward = self._evaluate_action(
                action, gaussians, reward_views, render_func, render_args, group_mapping, original_lrs, initial_state
            )

            if reward > best_reward:
                best_reward = reward
                best_action = action.clone()
                best_log_prob = log_prob.clone()

        return best_action, best_log_prob, best_reward, new_hidden

    def _evaluate_action(
        self,
        action: torch.Tensor,
        gaussians,
        reward_views: List,
        render_func: Callable,
        render_args: tuple,
        group_mapping: Dict[str, int],
        original_lrs: Dict[str, float],
        initial_state: dict,
    ) -> float:
        """Evaluate an action by running a short simulation"""

        # Restore initial state
        self._restore_model_state(gaussians, initial_state)

        # Apply learning rate scaling
        apply_lr_scaling(gaussians.optimizer, action, group_mapping, original_lrs)

        # Run short rollout (K steps)
        total_loss = 0.0

        for step in range(min(self.K, len(reward_views) * 3)):  # Limit steps
            # Pick reward view
            view_idx = step % len(reward_views)
            viewpoint_cam = reward_views[view_idx]
            pose = gaussians.get_RT(viewpoint_cam.uid)

            # Render and compute loss
            render_pkg = render_func(viewpoint_cam, gaussians, *render_args, camera_pose=pose)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()

            # Simple L1 loss for reward
            loss = torch.mean(torch.abs(image - gt_image))
            loss.backward()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

        # Restore original learning rates
        restore_optimizer_lrs(gaussians.optimizer, original_lrs)

        # Return negative loss as reward (higher is better)
        return -total_loss / min(self.K, len(reward_views) * 3)

    def _save_model_state(self, gaussians) -> dict:
        return {
            "model_state": {
                "_xyz": gaussians._xyz.data.clone(),
                "_features_dc": gaussians._features_dc.data.clone(),
                "_features_rest": gaussians._features_rest.data.clone(),
                "_scaling": gaussians._scaling.data.clone(),
                "_rotation": gaussians._rotation.data.clone(),
                "_opacity": gaussians._opacity.data.clone(),
                "P": gaussians.P.data.clone() if hasattr(gaussians, "P") else None,
            },
            "optimizer_state": copy.deepcopy(gaussians.optimizer.state_dict()),
        }

    def _restore_model_state(self, gaussians, saved_state: dict):
        model_state = saved_state["model_state"]

        gaussians._xyz.data.copy_(model_state["_xyz"])
        gaussians._features_dc.data.copy_(model_state["_features_dc"])
        gaussians._features_rest.data.copy_(model_state["_features_rest"])
        gaussians._scaling.data.copy_(model_state["_scaling"])
        gaussians._rotation.data.copy_(model_state["_rotation"])
        gaussians._opacity.data.copy_(model_state["_opacity"])

        if model_state["P"] is not None:
            gaussians.P.data.copy_(model_state["P"])

        gaussians.optimizer.load_state_dict(saved_state["optimizer_state"])

    def update_policy(
        self,
        policy,
        policy_optimizer: torch.optim.Optimizer,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        baseline: float,
        log_prob: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        """Update policy using REINFORCE with entropy bonus"""

        # Compute advantage
        advantage = reward - baseline

        # Policy loss: -log_prob * advantage
        policy_loss = -log_prob * advantage

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
            "advantage": advantage,
        }
