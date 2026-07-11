from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from grid_world.artifacts import imagined_rollout, load_dqn, load_world_model, real_rollout, write_sample_rollouts
from grid_world.config import GridConfig, RUNS_DIR
from grid_world.env import GridWorld
from grid_world.utils import write_json


def evaluate(
    world_model_path: Path = RUNS_DIR / "latest" / "world_model.pt",
    controller_path: Path = RUNS_DIR / "latest" / "controller.pt",
    output_dir: Path = RUNS_DIR / "latest",
    episodes: int = 25,
    seed: int = 1000,
    grid_config: GridConfig = GridConfig(),
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = load_world_model(world_model_path, device)
    dqn = load_dqn(controller_path, device)
    env = GridWorld(grid_config, random_start=True, seed=seed)

    real_returns: List[float] = []
    imagined_returns: List[float] = []
    drift_by_step: List[List[float]] = []
    samples = []

    for episode in range(episodes):
        start_seed = seed + episode
        start_state = env.reset(start_seed)
        real = real_rollout(GridWorld(grid_config, random_start=True, seed=seed), dqn, start_seed, device)
        imagined = imagined_rollout(world_model, dqn, start_state, grid_config.max_steps, device)
        real_returns.append(float(real["return"]))
        imagined_returns.append(float(imagined["return"]))
        drift_by_step.append(compute_drift(real["states"], imagined["states"]))
        if episode < 5:
            samples.append({"real": real, "imagined": imagined})

    drift = aggregate_drift(drift_by_step)
    payload = {
        "grid": env.spec(),
        "summary": {
            "episodes": episodes,
            "mean_real_return": float(np.mean(real_returns)),
            "mean_imagined_return": float(np.mean(imagined_returns)),
            "mean_return_gap": float(np.mean(np.asarray(imagined_returns) - np.asarray(real_returns))),
        },
        "drift": drift,
        "samples": samples,
    }
    write_json(output_dir / "evaluation.json", payload)
    write_sample_rollouts(payload, output_dir / "rollouts.json")
    return payload["summary"]


def compute_drift(real_states: List[List[float]], imagined_states: List[List[float]]) -> List[float]:
    horizon = min(len(real_states), len(imagined_states))
    values = []
    for idx in range(horizon):
        real = np.asarray(real_states[idx], dtype=np.float32)
        imagined = np.asarray(imagined_states[idx], dtype=np.float32)
        values.append(float(np.linalg.norm(real[:2] - imagined[:2])))
    return values


def aggregate_drift(runs: List[List[float]]) -> List[float]:
    max_len = max((len(run) for run in runs), default=0)
    result = []
    for idx in range(max_len):
        values = [run[idx] for run in runs if idx < len(run)]
        result.append(float(np.mean(values)))
    return result
