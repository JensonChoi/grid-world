import torch

from grid_world.models import GRUWorldModel, MLPWorldModel


def test_mlp_world_model_predict_step_shapes():
    model = MLPWorldModel(state_size=4, action_size=4, hidden_size=8)

    next_state, reward, done_prob = model.predict_step(torch.zeros(4), torch.tensor(1))

    assert next_state.shape == (4,)
    assert reward.shape == (1,)
    assert done_prob.shape == (1,)


def test_gru_world_model_forward_sequence_shapes():
    model = GRUWorldModel(state_size=4, action_size=4, hidden_size=8)

    next_states, rewards, done_logits = model(torch.zeros(2, 3, 4), torch.zeros(2, 3, dtype=torch.long))

    assert next_states.shape == (2, 3, 4)
    assert rewards.shape == (2, 3, 1)
    assert done_logits.shape == (2, 3, 1)


def test_gru_world_model_predict_step_shapes():
    model = GRUWorldModel(state_size=4, action_size=4, hidden_size=8)

    next_state, reward, done_prob = model.predict_step(torch.zeros(4), torch.tensor(1))

    assert next_state.shape == (4,)
    assert reward.shape == (1,)
    assert done_prob.shape == (1,)
