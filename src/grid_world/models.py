from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from grid_world.utils import one_hot


class MLPWorldModel(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.action_size = action_size
        self.backbone = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(hidden_size, state_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([states, one_hot(actions, self.action_size)], dim=-1)
        features = self.backbone(x)
        next_states = torch.sigmoid(self.next_state_head(features))
        rewards = self.reward_head(features)
        done_logits = self.done_head(features)
        return next_states, rewards, done_logits

    @torch.no_grad()
    def predict_step(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 0:
            action = action.unsqueeze(0)
        next_state, reward, done_logit = self.forward(state, action)
        done_prob = torch.sigmoid(done_logit)
        return next_state.squeeze(0), reward.squeeze(0), done_prob.squeeze(0)


class GRUWorldModel(nn.Module):
    """Optional sequence model kept as a comparison path for later experiments."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.action_size = action_size
        self.encoder = nn.Linear(state_size + action_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.next_state_head = nn.Linear(hidden_size, state_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        x = torch.cat([states, one_hot(actions, self.action_size)], dim=-1)
        encoded = torch.relu(self.encoder(x))
        output, _ = self.gru(encoded)
        return torch.sigmoid(self.next_state_head(output)), self.reward_head(output), self.done_head(output)

    @torch.no_grad()
    def predict_step(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 0:
            action = action.unsqueeze(0)
        next_state, reward, done_logit = self.forward(state, action)
        done_prob = torch.sigmoid(done_logit)
        return next_state.squeeze(0).squeeze(0), reward.squeeze(0).squeeze(0), done_prob.squeeze(0).squeeze(0)


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)
