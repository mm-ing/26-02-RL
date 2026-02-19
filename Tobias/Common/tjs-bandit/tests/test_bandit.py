import pytest
from TJS_bandit_logic import Bandit, Environment, EpsilonGreedyPolicy, ThompsonSamplingPolicy, Agent

def test_p_calculation():
    bandit = Bandit(start_coins=20)
    assert bandit.p == 0.2  # p should be 20% for 20 start coins

def test_pull_deterministic():
    bandit = Bandit(start_coins=20)
    bandit.pull()  # Pull the lever once
    assert bandit.clicks == 1  # Should have one click
    assert isinstance(bandit.last, int)  # Last payout should be an integer
    assert bandit.total >= 0  # Total should not be negative

def test_agent_step_updates_estimates():
    env = Environment(starts=(20, 40, 80))
    policy = EpsilonGreedyPolicy(epsilon=0.9)
    agent = Agent(env, policy)
    initial_estimates = [bandit.estimated_mean for bandit in env.bandits]
    
    agent.step()  # Perform one step
    updated_estimates = [bandit.estimated_mean for bandit in env.bandits]
    
    assert updated_estimates != initial_estimates  # Estimates should change after a step
    assert all(isinstance(est, float) for est in updated_estimates)  # Estimates should be floats