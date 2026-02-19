import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gridworld_logic import EpsilonScheduler, GridMap, MonteCarlo, QLearning


def test_grid_boundary_keeps_agent_in_place() -> None:
    grid = GridMap(rows=3, cols=3, blocked=[], start=(0, 0), target=(2, 2))
    assert grid.next_state((0, 0), 0) == (0, 0)  # up outside
    assert grid.next_state((0, 0), 2) == (0, 0)  # left outside


def test_blocked_transition_stays_in_place() -> None:
    grid = GridMap(rows=3, cols=3, blocked=[(1, 0)], start=(0, 0), target=(2, 2))
    assert grid.next_state((0, 0), 3) == (0, 0)  # right into blocked


def test_reward_function_correctness() -> None:
    grid = GridMap(rows=3, cols=3, blocked=[], start=(0, 0), target=(2, 2))
    assert grid.reward((2, 2)) == 0
    assert grid.reward((1, 2)) == -1


def test_q_learning_update_rule() -> None:
    q = QLearning(alpha=0.5, gamma=0.9)
    state = (0, 0)
    action = 3
    next_state = (1, 0)

    q.q_table[(state, action)] = 2.0
    q.q_table[(next_state, 0)] = 4.0
    q.q_table[(next_state, 1)] = 1.0
    q.q_table[(next_state, 2)] = 3.0
    q.q_table[(next_state, 3)] = 2.0

    reward = -1
    q.update(state, action, reward, next_state)

    expected = 2.0 + 0.5 * ((-1 + 0.9 * 4.0) - 2.0)
    assert q.q_table[(state, action)] == pytest.approx(expected)


def test_monte_carlo_first_visit_return_calculation() -> None:
    mc = MonteCarlo(gamma=1.0)
    trajectory = [
        ((0, 0), 3, (1, 0), -1, False),
        ((1, 0), 3, (2, 0), -1, False),
        ((2, 0), 1, (2, 1), 0, True),
    ]
    mc.finish_episode(trajectory)

    assert mc.value_table[(2, 0)] == pytest.approx(0.0)
    assert mc.value_table[(1, 0)] == pytest.approx(-1.0)
    assert mc.value_table[(0, 0)] == pytest.approx(-2.0)


def test_epsilon_decay_with_lower_bound() -> None:
    scheduler = EpsilonScheduler(epsilon_max=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    assert scheduler.value(0) == pytest.approx(1.0)
    assert scheduler.value(1) == pytest.approx(0.5)
    assert scheduler.value(2) == pytest.approx(0.25)
    assert scheduler.value(10) == pytest.approx(0.1)


def test_deterministic_transition_model() -> None:
    grid = GridMap(rows=3, cols=3, blocked=[(1, 1)], start=(0, 2), target=(2, 2))
    state = (1, 2)

    # into blocked
    assert grid.next_state(state, 0) == (1, 2)
    # valid move right
    assert grid.next_state(state, 3) == (2, 2)
