# Repository Guidelines

## Project Structure & Module Organization

This is a `uv`-managed Python project for an imagination-based gridworld agent.
Source code lives in `src/grid_world/`. Core modules include `env.py` for the
custom gridworld, `data.py` for transition collection, `models.py` for PyTorch
models, `train_world_model.py` and `train_controller.py` for training loops, and
`evaluate.py` for rollout comparison. The FastAPI web app is in `web.py`, with
static browser assets in `src/grid_world/static/`. Tests live in `tests/`.
Generated artifacts should go under `data/` or `runs/`, not in source modules.

## Build, Test, and Development Commands

- `uv sync --extra dev`: create/update the local environment with test tools.
- `uv run collect-data --episodes 1000`: collect random-policy transitions.
- `uv run train-world-model --epochs 30`: train the default MLP dynamics model.
- `uv run train-world-model --model-type gru --sequence-length 8 --epochs 30`:
  train the GRU dynamics model on contiguous rollout windows.
- `uv run benchmark-world-models --epochs 50`: compare MLP vs GRU validation
  error, open-loop drift, training time, and inference throughput with
  early-stopping convergence by default.
- `uv run train-controller --episodes 500`: train DQN inside the learned model.
- `uv run evaluate`: write evaluation and rollout artifacts to `runs/latest/`.
- `uv run serve`: start the FastAPI UI at `http://127.0.0.1:8000`.
- `uv run --extra dev python -m pytest`: run the test suite.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax with 4-space indentation and type hints for public
functions. Keep modules focused by responsibility, and prefer small dataclasses
for configuration objects. Use `snake_case` for functions, variables, files, and
CLI command implementation functions. Use `PascalCase` for classes such as
`GridWorld`, `MLPWorldModel`, `GRUWorldModel`, and `DQN`. Keep comments brief
and reserve them for non-obvious logic.

## Testing Guidelines

Tests use `pytest`. Place tests in `tests/` with names like `test_env.py` and
functions named `test_<behavior>()`. Add focused tests for environment rules,
dataset shape/schema, model smoke behavior, and evaluation output contracts.
Before submitting changes, run:

```bash
uv run --extra dev python -m pytest
```

## Commit & Pull Request Guidelines

This repository has no established Git history yet. Use short, imperative commit
messages such as `Add world model training loop` or `Fix rollout JSON endpoint`.
Pull requests should include a concise summary, test results, and any generated
artifact notes. For UI-facing changes, include a screenshot or describe the
manual browser check performed with `uv run serve`.

## Agent-Specific Instructions

Do not overwrite generated training artifacts unless the task calls for a new
run. Prefer changing source, tests, and docs separately from large files in
`data/` or `runs/`. GRU world-model training requires transition files collected
with the current data schema, including episode boundary metadata.
