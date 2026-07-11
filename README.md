# Imagination Gridworld

A beginner world-model project inspired by Ha & Schmidhuber: collect low-dimensional
gridworld transitions, train a learned dynamics model, then train a DQN controller
inside that learned model.

## Quickstart

```bash
uv sync --extra dev
uv run collect-data --episodes 1000
uv run train-world-model --epochs 30
uv run train-controller --episodes 500
uv run evaluate
uv run serve
```

Open the printed local URL to inspect real vs imagined rollouts and drift.

## Project Shape

- `src/grid_world/env.py`: deterministic numeric-state gridworld.
- `src/grid_world/data.py`: random-policy transition collection and datasets.
- `src/grid_world/models.py`: MLP world model, optional GRU world model, DQN.
- `src/grid_world/train_world_model.py`: supervised dynamics training.
- `src/grid_world/train_controller.py`: DQN trained only against the learned model.
- `src/grid_world/evaluate.py`: real vs imagined evaluation artifacts.
- `src/grid_world/web.py`: FastAPI app serving a vanilla Canvas UI.
