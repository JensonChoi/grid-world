from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from grid_world.config import GridConfig, RUNS_DIR
from grid_world.env import GridWorld
from grid_world.models import DQN, MLPWorldModel
from grid_world.utils import ensure_dir, write_json


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)


def load_world_model(path: Path, device: torch.device) -> MLPWorldModel:
    payload = torch.load(path, map_location=device)
    metadata = payload["metadata"]
    model = MLPWorldModel(
        state_size=int(metadata["state_size"]),
        action_size=int(metadata["action_size"]),
        hidden_size=int(metadata["hidden_size"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def load_dqn(path: Path, device: torch.device) -> DQN:
    payload = torch.load(path, map_location=device)
    metadata = payload["metadata"]
    model = DQN(
        state_size=int(metadata["state_size"]),
        action_size=int(metadata["action_size"]),
        hidden_size=int(metadata["hidden_size"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def trajectory_to_json(states: List[np.ndarray], actions: List[int], rewards: List[float]) -> Dict[str, Any]:
    return {
        "states": [state.tolist() for state in states],
        "actions": actions,
        "rewards": rewards,
        "return": float(np.sum(rewards)) if rewards else 0.0,
        "length": len(actions),
    }


@torch.no_grad()
def imagined_rollout(
    world_model: MLPWorldModel,
    dqn: DQN,
    start_state: np.ndarray,
    horizon: int,
    device: torch.device,
) -> Dict[str, Any]:
    state = torch.tensor(start_state, dtype=torch.float32, device=device)
    states = [start_state.astype(np.float32)]
    actions: List[int] = []
    rewards: List[float] = []
    for _ in range(horizon):
        action = int(torch.argmax(dqn(state.unsqueeze(0)), dim=-1).item())
        next_state, reward, done_prob = world_model.predict_step(state, torch.tensor(action, device=device))
        actions.append(action)
        rewards.append(float(reward.item()))
        state = next_state.clamp(0.0, 1.0)
        states.append(state.detach().cpu().numpy().astype(np.float32))
        if float(done_prob.item()) > 0.5:
            break
    return trajectory_to_json(states, actions, rewards)


def real_rollout(env: GridWorld, dqn: DQN, start_seed: int, device: torch.device) -> Dict[str, Any]:
    state = env.reset(start_seed)
    states = [state]
    actions: List[int] = []
    rewards: List[float] = []
    for _ in range(env.config.max_steps):
        with torch.no_grad():
            tensor_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(torch.argmax(dqn(tensor_state), dim=-1).item())
        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        states.append(state)
        if done:
            break
    return trajectory_to_json(states, actions, rewards)


def write_sample_rollouts(payload: Dict[str, Any], output: Path = RUNS_DIR / "latest" / "rollouts.json") -> Path:
    write_json(output, payload)
    return output
