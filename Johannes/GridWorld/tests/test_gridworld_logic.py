import sys
import os
import pytest

# ensure GridWorld package path is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gridworld_logic import Grid, QLearningAgent, Trainer


def test_step_and_reward():
    g = Grid(M=3, N=3, blocked=[(1, 0)], start=(0, 0), target=(2, 2))
    s = (0, 0)
    # move right (action 3)
    s2, r, done = g.step(s, 3)
    assert isinstance(s2, tuple)


def test_blocked_enforcement():
    g = Grid(M=3, N=3, blocked=[(1, 0)], start=(0, 0), target=(2, 2))
    s = (0, 0)
    # moving right into blocked should not return blocked cell
    s2, r, done = g.step(s, 3)
    assert s2 != (1, 0)


def test_q_update():
    g = Grid(M=3, N=3, blocked=[], start=(0, 0), target=(2, 2))
    q = QLearningAgent(g, alpha=1.0, gamma=0.0)
    s = (0, 0)
    a = 3
    s2, r, done = g.step(s, a)
    q.update(s, a, r, s2)
    assert q.get_q(s, a) == pytest.approx(r)
