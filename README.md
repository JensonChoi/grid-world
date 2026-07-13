# Imagination Gridworld

A beginner world-model project inspired by Ha & Schmidhuber: collect low-dimensional
gridworld transitions, train a learned dynamics model, then train a DQN controller
inside that learned model.

## Quickstart

```bash
uv sync --extra dev
uv run collect-data --episodes 1000
uv run train-world-model --epochs 30
uv run train-world-model --model-type gru --sequence-length 8 --epochs 30
uv run train-controller --episodes 500
uv run evaluate
uv run serve
```

Open the printed local URL to inspect real vs imagined rollouts and drift.

`train-world-model` uses an MLP by default. Pass `--model-type gru` to train the
sequence model on contiguous rollout windows. GRU training requires data
collected by the current `collect-data` command because older transition files
do not include episode boundary metadata.

## Project Shape

- `src/grid_world/env.py`: deterministic numeric-state gridworld.
- `src/grid_world/data.py`: random-policy transition collection and datasets.
- `src/grid_world/models.py`: MLP world model, optional GRU world model, DQN.
- `src/grid_world/train_world_model.py`: supervised dynamics training.
- `src/grid_world/train_controller.py`: DQN trained only against the learned model.
- `src/grid_world/evaluate.py`: real vs imagined evaluation artifacts.
- `src/grid_world/web.py`: FastAPI app serving a vanilla Canvas UI.
