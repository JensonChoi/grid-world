from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from grid_world.config import GridConfig


Action = int


class GridWorld:
    """Small deterministic gridworld with exact low-dimensional states."""

    ACTIONS: Dict[Action, Tuple[int, int]] = {
        0: (0, -1),
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0),
    }

    def __init__(self, config: Optional[GridConfig] = None, random_start: bool = True, seed: int = 0):
        self.config = config or GridConfig()
        self.random_start = random_start
        self.rng = np.random.default_rng(seed)
        self.wall_set = set(self.config.walls)
        self.pos = self.config.start
        self.steps = 0

    @property
    def observation_size(self) -> int:
        return 4

    @property
    def action_size(self) -> int:
        return len(self.ACTIONS)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        if self.random_start:
            free = [
                (x, y)
                for y in range(self.config.height)
                for x in range(self.config.width)
                if (x, y) not in self.wall_set and (x, y) != self.config.goal
            ]
            self.pos = free[int(self.rng.integers(0, len(free)))]
        else:
            self.pos = self.config.start
        return self.state()

    def state(self) -> np.ndarray:
        width = max(1, self.config.width - 1)
        height = max(1, self.config.height - 1)
        return np.array(
            [
                self.pos[0] / width,
                self.pos[1] / height,
                self.config.goal[0] / width,
                self.config.goal[1] / height,
            ],
            dtype=np.float32,
        )

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action {action}")

        self.steps += 1
        dx, dy = self.ACTIONS[action]
        candidate = (self.pos[0] + dx, self.pos[1] + dy)
        hit_wall = False
        if self._is_valid(candidate):
            self.pos = candidate
        else:
            hit_wall = True

        reached_goal = self.pos == self.config.goal
        truncated = self.steps >= self.config.max_steps and not reached_goal
        done = reached_goal or truncated
        reward = 1.0 if reached_goal else -0.01
        if hit_wall:
            reward -= 0.04

        return self.state(), reward, done, {"hit_wall": hit_wall, "truncated": truncated}

    def clone_at(self, state: np.ndarray) -> "GridWorld":
        env = GridWorld(self.config, random_start=False)
        env.pos = self.decode_agent_position(state)
        env.steps = self.steps
        return env

    def decode_agent_position(self, state: np.ndarray) -> Tuple[int, int]:
        x = int(round(float(state[0]) * (self.config.width - 1)))
        y = int(round(float(state[1]) * (self.config.height - 1)))
        return (int(np.clip(x, 0, self.config.width - 1)), int(np.clip(y, 0, self.config.height - 1)))

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.config.width and 0 <= y < self.config.height and pos not in self.wall_set

    def spec(self) -> Dict[str, object]:
        payload = asdict(self.config)
        payload["actions"] = {str(k): v for k, v in self.ACTIONS.items()}
        return payload


def shortest_path_policy(env: GridWorld, state: np.ndarray) -> int:
    """Greedy one-step lookahead policy for reference evaluation."""
    pos = env.decode_agent_position(state)
    goal = env.config.goal
    best_action = 0
    best_dist = 10**9
    for action, (dx, dy) in env.ACTIONS.items():
        candidate = (pos[0] + dx, pos[1] + dy)
        if not env._is_valid(candidate):
            candidate = pos
        dist = abs(candidate[0] - goal[0]) + abs(candidate[1] - goal[1])
        if dist < best_dist:
            best_action = action
            best_dist = dist
    return best_action
