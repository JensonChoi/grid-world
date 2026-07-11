from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"


@dataclass(frozen=True)
class GridConfig:
    width: int = 8
    height: int = 8
    max_steps: int = 40
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (7, 7)
    walls: List[Tuple[int, int]] = field(
        default_factory=lambda: [(2, 2), (2, 3), (2, 4), (5, 3), (5, 4), (5, 5)]
    )


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    batch_size: int = 128
    hidden_size: int = 128
    learning_rate: float = 1e-3
