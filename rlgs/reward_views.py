import torch
import numpy as np
from typing import List, Tuple
from random import shuffle


class RewardViewSampler:
    """
    Manages reward view sampling for RLGS.
    Holds out a small subset of views for policy evaluation.
    """

    def __init__(
        self,
        train_cameras: List,
        reward_set_len: int = 2,
        reshuffle_interval: int = 100,
    ):
        """
        Args:
            train_cameras: List of training cameras
            reward_set_len: Number of views to hold out for reward computation
            reshuffle_interval: How often to reshuffle reward views (iterations)
        """
        self.all_cameras = train_cameras.copy()
        self.reward_set_len = reward_set_len
        self.reshuffle_interval = reshuffle_interval

        self.reward_views = []
        self.train_views = []
        self.last_reshuffle = 0

        self._reshuffle_views()

    def _reshuffle_views(self):
        """Randomly select new reward views"""
        shuffled_cameras = self.all_cameras.copy()
        shuffle(shuffled_cameras)

        self.reward_views = shuffled_cameras[: self.reward_set_len]
        self.train_views = shuffled_cameras[self.reward_set_len :]

        print(f"Reshuffled reward views: {len(self.reward_views)} reward, {len(self.train_views)} train")

    def maybe_reshuffle(self, current_iteration: int):
        """Check if we should reshuffle views based on iteration"""
        if current_iteration - self.last_reshuffle >= self.reshuffle_interval:
            self._reshuffle_views()
            self.last_reshuffle = current_iteration

    def get_reward_views(self) -> List:
        """Get current reward views for evaluation"""
        return self.reward_views.copy()

    def get_train_views(self) -> List:
        """Get current training views"""
        return self.train_views.copy()

    def get_random_train_view(self):
        """Get a random training view (excludes reward views)"""
        if not self.train_views:
            return self.all_cameras[0]  # Fallback

        idx = np.random.randint(len(self.train_views))

        return self.train_views[idx]
