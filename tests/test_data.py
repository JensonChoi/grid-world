import numpy as np
import pytest

from grid_world.config import GridConfig
from grid_world.data import SequenceTransitionDataset, collect_random_transitions


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
    assert data["episode_starts"].shape == (3,)
    assert data["episode_lengths"].shape == (3,)
    assert result["transitions"] <= 15


def test_sequence_transition_dataset_uses_episode_windows(tmp_path):
    output = tmp_path / "transitions.npz"
    collect_random_transitions(
        episodes=2,
        seed=1,
        config=GridConfig(width=4, height=4, max_steps=4, goal=(3, 3), walls=[]),
        output=output,
    )

    dataset = SequenceTransitionDataset(output, sequence_length=2)

    states, actions, next_states, rewards, dones = dataset[0]
    assert states.shape == (2, 4)
    assert actions.shape == (2,)
    assert next_states.shape == (2, 4)
    assert rewards.shape == (2, 1)
    assert dones.shape == (2, 1)
    for start, end in dataset.windows:
        assert end - start == 2


def test_sequence_transition_dataset_requires_episode_metadata(tmp_path):
    output = tmp_path / "old_transitions.npz"
    np.savez_compressed(
        output,
        states=np.zeros((2, 4), dtype=np.float32),
        actions=np.zeros(2, dtype=np.int64),
        next_states=np.zeros((2, 4), dtype=np.float32),
        rewards=np.zeros(2, dtype=np.float32),
        dones=np.zeros(2, dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="Re-run collect-data"):
        SequenceTransitionDataset(output, sequence_length=2)
