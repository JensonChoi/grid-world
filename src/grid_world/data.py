from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from grid_world.config import DATA_DIR, GridConfig
from grid_world.env import GridWorld
from grid_world.utils import ensure_dir


@dataclass
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


class TransitionDataset(Dataset):
    def __init__(self, path: Path):
        data = np.load(path)
        self.states = torch.tensor(data["states"], dtype=torch.float32)
        self.actions = torch.tensor(data["actions"], dtype=torch.long)
        self.next_states = torch.tensor(data["next_states"], dtype=torch.float32)
        self.rewards = torch.tensor(data["rewards"], dtype=torch.float32).unsqueeze(-1)
        self.dones = torch.tensor(data["dones"], dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            self.rewards[idx],
            self.dones[idx],
        )


def collect_random_transitions(
    episodes: int,
    seed: int,
    config: GridConfig,
    output: Path = DATA_DIR / "random_transitions.npz",
) -> Dict[str, object]:
    env = GridWorld(config=config, random_start=True, seed=seed)
    rng = np.random.default_rng(seed)
    states: List[np.ndarray] = []
    actions: List[int] = []
    next_states: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []
    episode_returns: List[float] = []
    episode_lengths: List[int] = []

    for episode in range(episodes):
        state = env.reset(seed + episode)
        total_reward = 0.0
        for step in range(config.max_steps):
            action = int(rng.integers(0, env.action_size))
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(float(reward))
            dones.append(bool(done))
            total_reward += float(reward)
            state = next_state
            if done:
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(config.max_steps)
        episode_returns.append(total_reward)

    ensure_dir(output.parent)
    np.savez_compressed(
        output,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        next_states=np.asarray(next_states, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
    )
    return {
        "path": str(output),
        "transitions": len(actions),
        "episodes": episodes,
        "mean_return": float(np.mean(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
    }
