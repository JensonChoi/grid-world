from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from grid_world.artifacts import load_world_model, save_checkpoint
from grid_world.config import GridConfig, RUNS_DIR, TrainConfig
from grid_world.env import GridWorld
from grid_world.models import DQN
from grid_world.utils import set_seed, write_json


Experience = Tuple[np.ndarray, int, float, np.ndarray, bool]


def train_controller(
    world_model_path: Path,
    output_dir: Path = RUNS_DIR / "latest",
    episodes: int = 500,
    config: TrainConfig = TrainConfig(),
    grid_config: GridConfig = GridConfig(),
) -> Dict[str, object]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = load_world_model(world_model_path, device)
    for param in world_model.parameters():
        param.requires_grad_(False)

    env_template = GridWorld(grid_config, random_start=True, seed=config.seed)
    dqn = DQN(env_template.observation_size, env_template.action_size, config.hidden_size).to(device)
    target = DQN(env_template.observation_size, env_template.action_size, config.hidden_size).to(device)
    target.load_state_dict(dqn.state_dict())
    optimizer = torch.optim.Adam(dqn.parameters(), lr=config.learning_rate)
    replay: Deque[Experience] = deque(maxlen=20_000)
    rng = np.random.default_rng(config.seed)
    gamma = 0.97
    epsilon_start = 1.0
    epsilon_end = 0.05
    returns: List[float] = []

    for episode in range(episodes):
        state = env_template.reset(config.seed + episode)
        episode_return = 0.0
        epsilon = max(epsilon_end, epsilon_start - episode / max(1, episodes * 0.7))
        for _ in range(grid_config.max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(0, env_template.action_size))
            else:
                with torch.no_grad():
                    q_values = dqn(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = int(torch.argmax(q_values, dim=-1).item())

            next_state, reward, done = imagined_step(world_model, state, action, device)
            replay.append((state, action, reward, next_state, done))
            episode_return += reward
            state = next_state

            if len(replay) >= config.batch_size:
                update_dqn(dqn, target, replay, optimizer, config.batch_size, gamma, device, rng)
            if done:
                break

        if episode % 20 == 0:
            target.load_state_dict(dqn.state_dict())
        returns.append(float(episode_return))

    checkpoint = output_dir / "controller.pt"
    metadata = {
        "state_size": env_template.observation_size,
        "action_size": env_template.action_size,
        "hidden_size": config.hidden_size,
        "episodes": episodes,
        "world_model_path": str(world_model_path),
    }
    save_checkpoint(checkpoint, dqn, metadata)
    write_json(output_dir / "controller_metrics.json", {"returns": returns, "metadata": metadata})
    return {"checkpoint": str(checkpoint), "mean_imagined_return": float(np.mean(returns[-50:])), "device": str(device)}


@torch.no_grad()
def imagined_step(world_model, state: np.ndarray, action: int, device: torch.device) -> Tuple[np.ndarray, float, bool]:
    tensor_state = torch.tensor(state, dtype=torch.float32, device=device)
    tensor_action = torch.tensor(action, dtype=torch.long, device=device)
    next_state, reward, done_prob = world_model.predict_step(tensor_state, tensor_action)
    return (
        next_state.clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32),
        float(reward.item()),
        bool(float(done_prob.item()) > 0.5),
    )


def update_dqn(
    dqn: DQN,
    target: DQN,
    replay: Deque[Experience],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    idx = rng.choice(len(replay), size=batch_size, replace=False)
    batch = [replay[int(i)] for i in idx]
    states = torch.tensor(np.asarray([b[0] for b in batch]), dtype=torch.float32, device=device)
    actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device).unsqueeze(-1)
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
    next_states = torch.tensor(np.asarray([b[3] for b in batch]), dtype=torch.float32, device=device)
    dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

    q_values = dqn(states).gather(1, actions).squeeze(-1)
    with torch.no_grad():
        next_q = target(next_states).max(dim=-1).values
        expected = rewards + gamma * (1.0 - dones) * next_q
    loss = nn.functional.smooth_l1_loss(q_values, expected)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
