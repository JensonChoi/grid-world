from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from grid_world.artifacts import save_checkpoint
from grid_world.config import RUNS_DIR, TrainConfig
from grid_world.data import SequenceTransitionDataset, TransitionDataset
from grid_world.models import GRUWorldModel, MLPWorldModel
from grid_world.utils import ensure_dir, set_seed, write_json


def train_world_model(
    data_path: Path,
    output_dir: Path = RUNS_DIR / "latest",
    epochs: int = 30,
    config: TrainConfig = TrainConfig(),
    model_type: Literal["mlp", "gru"] = "mlp",
    sequence_length: int = 8,
) -> Dict[str, object]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = (
        SequenceTransitionDataset(data_path, sequence_length)
        if model_type == "gru"
        else TransitionDataset(data_path)
    )
    train_size = max(1, int(0.85 * len(dataset)))
    val_size = max(1, len(dataset) - train_size)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    state_size = int(dataset.states.shape[-1])
    action_size = int(dataset.actions.max().item()) + 1
    if model_type == "gru":
        model = GRUWorldModel(state_size, action_size, config.hidden_size).to(device)
    elif model_type == "mlp":
        model = MLPWorldModel(state_size, action_size, config.hidden_size).to(device)
    else:
        raise ValueError(f"Unsupported world model type: {model_type}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_state_mse": [], "val_reward_mse": [], "val_done_bce": []}
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for states, actions, next_states, rewards, dones in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)
            pred_next, pred_reward, pred_done = model(states, actions)
            loss = mse(pred_next, next_states) + mse(pred_reward, rewards) + 0.2 * bce(pred_done, dones)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * states.shape[0]
            count += states.shape[0]

        val_metrics = evaluate_world_model(model, val_loader, device)
        history["train_loss"].append(total / max(1, count))
        for key, value in val_metrics.items():
            history[key].append(value)

    ensure_dir(output_dir)
    checkpoint = output_dir / "world_model.pt"
    metadata = {
        "state_size": state_size,
        "action_size": action_size,
        "hidden_size": config.hidden_size,
        "epochs": epochs,
        "data_path": str(data_path),
        "model_type": model_type,
        "sequence_length": sequence_length if model_type == "gru" else None,
    }
    save_checkpoint(checkpoint, model, metadata)
    write_json(output_dir / "world_model_metrics.json", {"history": history, "metadata": metadata})
    save_loss_plot(history, output_dir / "world_model_loss.png")
    return {"checkpoint": str(checkpoint), "metrics": history, "device": str(device)}


@torch.no_grad()
def evaluate_world_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    totals = {"val_state_mse": 0.0, "val_reward_mse": 0.0, "val_done_bce": 0.0}
    count = 0
    for states, actions, next_states, rewards, dones in loader:
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        pred_next, pred_reward, pred_done = model(states, actions)
        totals["val_state_mse"] += float(mse(pred_next, next_states).item())
        totals["val_reward_mse"] += float(mse(pred_reward, rewards).item())
        totals["val_done_bce"] += float(bce(pred_done, dones).item())
        count += states.shape[0]
    return {key: value / max(1, count) for key, value in totals.items()}


def save_loss_plot(history: Dict[str, list], output: Path) -> None:
    ensure_dir(output.parent)
    plt.figure(figsize=(8, 4))
    for key, values in history.items():
        plt.plot(np.arange(1, len(values) + 1), values, label=key)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
