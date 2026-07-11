import numpy as np

from grid_world.config import GridConfig
from grid_world.env import GridWorld


def test_wall_collision_keeps_agent_in_place():
    env = GridWorld(GridConfig(width=4, height=4, start=(1, 0), goal=(3, 3), walls=[(1, 1)]), random_start=False)
    state = env.reset()
    next_state, reward, done, info = env.step(2)
    assert np.allclose(state[:2], next_state[:2])
    assert reward < -0.01
    assert done is False
    assert info["hit_wall"] is True


def test_goal_ends_episode_with_positive_reward():
    env = GridWorld(GridConfig(width=3, height=3, start=(1, 2), goal=(2, 2), walls=[]), random_start=False)
    env.reset()
    _, reward, done, _ = env.step(1)
    assert done is True
    assert reward == 1.0


def test_state_contains_normalized_agent_and_goal_positions():
    env = GridWorld(GridConfig(width=5, height=5, start=(0, 0), goal=(4, 4), walls=[]), random_start=False)
    state = env.reset()
    assert state.tolist() == [0.0, 0.0, 1.0, 1.0]
