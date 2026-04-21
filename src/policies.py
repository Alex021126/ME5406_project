from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SplitObservationFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, joint_count: int):
        super().__init__(observation_space, features_dim=256)
        self._joint_count = joint_count
        self._proprio_dim = 2 * joint_count
        self._task_dim = 4
        self._obstacle_dim = 3

        self.proprio_branch = nn.Sequential(
            nn.Linear(self._proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.context_branch = nn.Sequential(
            nn.Linear(self._task_dim + self._obstacle_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        proprio = observations[:, : self._proprio_dim]
        task_context = observations[:, self._proprio_dim : self._proprio_dim + self._task_dim]
        obstacle = observations[:, -self._obstacle_dim :]
        proprio_feature = self.proprio_branch(proprio)
        context_feature = self.context_branch(torch.cat([task_context, obstacle], dim=1))
        return self.fusion(torch.cat([proprio_feature, context_feature], dim=1))


class GoalConditionedSplitFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, joint_count: int):
        super().__init__(observation_space, features_dim=256)
        self._joint_count = joint_count
        self._proprio_dim = 2 * joint_count
        self._task_dim = 4
        self._obstacle_dim = 3

        self.proprio_branch = nn.Sequential(
            nn.Linear(self._proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.context_branch = nn.Sequential(
            nn.Linear(self._task_dim + self._obstacle_dim + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        base_obs = observations["observation"]
        proprio = base_obs[:, : self._proprio_dim]
        task_context = base_obs[:, self._proprio_dim : self._proprio_dim + self._task_dim]
        obstacle = base_obs[:, -self._obstacle_dim :]
        goal_delta = observations["desired_goal"] - observations["achieved_goal"]
        proprio_feature = self.proprio_branch(proprio)
        context_feature = self.context_branch(torch.cat([task_context, obstacle, goal_delta], dim=1))
        return self.fusion(torch.cat([proprio_feature, context_feature], dim=1))
