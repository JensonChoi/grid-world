from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def one_hot(actions: torch.Tensor, action_size: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(actions.long(), num_classes=action_size).float()
