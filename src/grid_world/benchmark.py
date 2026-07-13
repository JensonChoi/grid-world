from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from grid_world.artifacts import load_world_model
from grid_world.config import DATA_DIR, RUNS_DIR, GridConfig, TrainConfig
from grid_world.env import GridWorld
from grid_world.train_world_model import train_world_model
from grid_world.utils import ensure_dir, set_seed, write_json


MODEL_TYPES = ("mlp", "gru")


def benchmark_world_models(
    data_path: Path = DATA_DIR / "random_transitions.npz",
    output_dir: Path = RUNS_DIR / "benchmarks",
    epochs: int = 10,
    sequence_length: int = 8,
    rollouts: int = 25,
    horizon: int = 20,
    seed: int = 2024,
    train_config: TrainConfig = TrainConfig(),
    grid_config: GridConfig = GridConfig(),
) -> Dict[str, object]:
    set_seed(seed)
    ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict[str, object] = {
        "metadata": {
            "data_path": str(data_path),
            "epochs": epochs,
            "sequence_length": sequence_length,
            "rollouts": rollouts,
            "horizon": horizon,
            "seed": seed,
            "device": str(device),
        },
        "models": {},
    }

    for model_type in MODEL_TYPES:
        model_dir = output_dir / model_type
        started = time.perf_counter()
        training = train_world_model(
            data_path=data_path,
            output_dir=model_dir,
            epochs=epochs,
            config=train_config,
            model_type=model_type,
            sequence_length=sequence_length,
        )
        train_seconds = time.perf_counter() - started
        model = load_world_model(Path(str(training["checkpoint"])), device)
        history = training["metrics"]
        drift = measure_open_loop_drift(model, rollouts, horizon, seed, grid_config, device)
        throughput = measure_inference_throughput(model, grid_config, device)
        results["models"][model_type] = {
            "checkpoint": training["checkpoint"],
            "train_seconds": train_seconds,
            "inference_steps_per_second": throughput,
            "final_val_state_mse": float(history["val_state_mse"][-1]),
            "final_val_reward_mse": float(history["val_reward_mse"][-1]),
            "final_val_done_bce": float(history["val_done_bce"][-1]),
            "mean_open_loop_drift": float(np.mean(drift)) if drift else 0.0,
            "open_loop_drift": drift,
        }

    plot_paths = save_benchmark_plots(results, output_dir)
    results["plots"] = {name: str(path) for name, path in plot_paths.items()}
    write_json(output_dir / "benchmarks.json", results)
    return results


@torch.no_grad()
def measure_open_loop_drift(
    model: torch.nn.Module,
    rollouts: int,
    horizon: int,
    seed: int,
    grid_config: GridConfig,
    device: torch.device,
) -> List[float]:
    rng = np.random.default_rng(seed)
    runs: List[List[float]] = []
    for episode in range(rollouts):
        env = GridWorld(grid_config, random_start=True, seed=seed + episode)
        real_state = env.reset(seed + episode)
        imagined_state = torch.tensor(real_state, dtype=torch.float32, device=device)
        errors: List[float] = []
        for _ in range(horizon):
            action = int(rng.integers(0, env.action_size))
            real_state, _, done, _ = env.step(action)
            next_state, _, _ = model.predict_step(imagined_state, torch.tensor(action, device=device))
            imagined_state = next_state.clamp(0.0, 1.0)
            imagined_np = imagined_state.detach().cpu().numpy()
            errors.append(float(np.linalg.norm(real_state[:2] - imagined_np[:2])))
            if done:
                break
        runs.append(errors)
    return aggregate_ragged(runs)


@torch.no_grad()
def measure_inference_throughput(
    model: torch.nn.Module,
    grid_config: GridConfig,
    device: torch.device,
    steps: int = 1_000,
) -> float:
    env = GridWorld(grid_config, random_start=False)
    state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    action = torch.tensor(0, dtype=torch.long, device=device)
    for _ in range(10):
        state, _, _ = model.predict_step(state, action)
        state = state.clamp(0.0, 1.0)

    started = time.perf_counter()
    for step in range(steps):
        action = torch.tensor(step % env.action_size, dtype=torch.long, device=device)
        state, _, _ = model.predict_step(state, action)
        state = state.clamp(0.0, 1.0)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return float(steps / elapsed)


def aggregate_ragged(runs: List[List[float]]) -> List[float]:
    max_len = max((len(run) for run in runs), default=0)
    return [
        float(np.mean([run[idx] for run in runs if idx < len(run)]))
        for idx in range(max_len)
    ]


def save_benchmark_plots(results: Dict[str, object], output_dir: Path) -> Dict[str, Path]:
    ensure_dir(output_dir)
    models = results["models"]
    model_names = list(MODEL_TYPES)

    validation_path = output_dir / "validation_mse.png"
    plt.figure(figsize=(7, 4))
    plt.bar(model_names, [models[name]["final_val_state_mse"] for name in model_names])
    plt.ylabel("final validation state MSE")
    plt.title("World Model Validation Error")
    plt.tight_layout()
    plt.savefig(validation_path)
    plt.close()

    drift_path = output_dir / "open_loop_drift.png"
    plt.figure(figsize=(7, 4))
    for name in model_names:
        drift = models[name]["open_loop_drift"]
        plt.plot(np.arange(1, len(drift) + 1), drift, marker="o", label=name.upper())
    plt.xlabel("open-loop step")
    plt.ylabel("mean agent-position error")
    plt.title("Open-Loop Drift Under Random Actions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(drift_path)
    plt.close()

    return {"validation_mse": validation_path, "open_loop_drift": drift_path}
