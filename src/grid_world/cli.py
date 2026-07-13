from __future__ import annotations

import argparse
from pathlib import Path

from grid_world.config import DATA_DIR, RUNS_DIR, GridConfig, TrainConfig
from grid_world.benchmark import benchmark_world_models
from grid_world.data import collect_random_transitions
from grid_world.evaluate import evaluate
from grid_world.train_controller import train_controller
from grid_world.train_world_model import train_world_model


def collect_data_main() -> None:
    parser = argparse.ArgumentParser(description="Collect random-policy gridworld transitions.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "random_transitions.npz")
    args = parser.parse_args()
    result = collect_random_transitions(args.episodes, args.seed, GridConfig(), args.output)
    print(result)


def train_world_model_main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned world model.")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "random_transitions.npz")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "latest")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-type", choices=["mlp", "gru"], default="mlp")
    parser.add_argument("--sequence-length", type=int, default=8)
    args = parser.parse_args()
    result = train_world_model(
        args.data,
        args.output_dir,
        args.epochs,
        TrainConfig(seed=args.seed),
        args.model_type,
        args.sequence_length,
    )
    print({"checkpoint": result["checkpoint"], "device": result["device"]})


def train_controller_main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN entirely inside the learned world model.")
    parser.add_argument("--world-model", type=Path, default=RUNS_DIR / "latest" / "world_model.pt")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "latest")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = train_controller(args.world_model, args.output_dir, args.episodes, TrainConfig(seed=args.seed))
    print(result)


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate imagined controller in real and imagined worlds.")
    parser.add_argument("--world-model", type=Path, default=RUNS_DIR / "latest" / "world_model.pt")
    parser.add_argument("--controller", type=Path, default=RUNS_DIR / "latest" / "controller.pt")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "latest")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    result = evaluate(args.world_model, args.controller, args.output_dir, args.episodes, args.seed)
    print(result)


def benchmark_world_models_main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLP and GRU world models.")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "random_transitions.npz")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "benchmarks")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    result = benchmark_world_models(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        sequence_length=args.sequence_length,
        rollouts=args.rollouts,
        horizon=args.horizon,
        seed=args.seed,
    )
    print({"output": str(args.output_dir), "plots": result["plots"]})
