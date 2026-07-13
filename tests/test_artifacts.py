import torch

from grid_world.artifacts import load_world_model, save_checkpoint
from grid_world.models import GRUWorldModel, MLPWorldModel


def test_load_world_model_dispatches_mlp_checkpoint(tmp_path):
    path = tmp_path / "mlp.pt"
    model = MLPWorldModel(state_size=4, action_size=4, hidden_size=8)
    save_checkpoint(path, model, {"state_size": 4, "action_size": 4, "hidden_size": 8, "model_type": "mlp"})

    loaded = load_world_model(path, torch.device("cpu"))

    assert isinstance(loaded, MLPWorldModel)


def test_load_world_model_dispatches_gru_checkpoint(tmp_path):
    path = tmp_path / "gru.pt"
    model = GRUWorldModel(state_size=4, action_size=4, hidden_size=8)
    save_checkpoint(path, model, {"state_size": 4, "action_size": 4, "hidden_size": 8, "model_type": "gru"})

    loaded = load_world_model(path, torch.device("cpu"))

    assert isinstance(loaded, GRUWorldModel)
