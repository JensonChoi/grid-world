import numpy as np

from grid_world.config import GridConfig
from grid_world.data import collect_random_transitions


def test_collect_random_transitions_writes_npz(tmp_path):
    output = tmp_path / "transitions.npz"
    result = collect_random_transitions(
        episodes=3,
        seed=1,
        config=GridConfig(width=4, height=4, max_steps=5, goal=(3, 3), walls=[]),
        output=output,
    )
    assert output.exists()
    data = np.load(output)
    assert data["states"].shape[1] == 4
    assert data["actions"].ndim == 1
    assert result["transitions"] <= 15
