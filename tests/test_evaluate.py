from grid_world.evaluate import aggregate_drift, compute_drift


def test_compute_drift_uses_agent_coordinates_only():
    real = [[0.0, 0.0, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]]
    imagined = [[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
    drift = compute_drift(real, imagined)
    assert drift == [0.0, 0.5]


def test_aggregate_drift_handles_ragged_runs():
    drift = aggregate_drift([[0.0, 1.0], [0.0]])
    assert drift == [0.0, 1.0]
