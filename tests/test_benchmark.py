from grid_world.benchmark import aggregate_ragged, benchmark_world_models
from grid_world.config import GridConfig, TrainConfig
from grid_world.data import collect_random_transitions


def test_aggregate_ragged_averages_available_steps():
    assert aggregate_ragged([[1.0, 3.0], [2.0]]) == [1.5, 3.0]


def test_benchmark_world_models_writes_metrics_and_plots(tmp_path):
    data_path = tmp_path / "transitions.npz"
    output_dir = tmp_path / "benchmarks"
    collect_random_transitions(
        episodes=3,
        seed=1,
        config=GridConfig(width=4, height=4, max_steps=4, goal=(3, 3), walls=[]),
        output=data_path,
    )

    result = benchmark_world_models(
        data_path=data_path,
        output_dir=output_dir,
        epochs=1,
        sequence_length=2,
        rollouts=2,
        horizon=3,
        seed=1,
        early_stopping_patience=1,
        train_config=TrainConfig(seed=1, batch_size=2, hidden_size=8),
        grid_config=GridConfig(width=4, height=4, max_steps=4, goal=(3, 3), walls=[]),
    )

    assert set(result["models"].keys()) == {"mlp", "gru"}
    assert (output_dir / "benchmarks.json").exists()
    assert (output_dir / "validation_mse.png").exists()
    assert (output_dir / "open_loop_drift.png").exists()
    assert result["models"]["mlp"]["epochs_trained"] >= 1
    assert result["models"]["gru"]["epochs_trained"] >= 1
