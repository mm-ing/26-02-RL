"""Core bandit environment, agent, and policy implementations.

Classes:
 - Environment: Bernoulli bandits with fixed probabilities.
 - Agent: interacts with Environment, holds memory and stats.
 - Policies: EpsilonGreedyPolicy and ThompsonSamplingPolicy.

Memory semantics: memory_size=100 (default). If memory_size == 0,
the agent treats it as unlimited (full memory).
"""

import random
from typing import List, Optional, Tuple


class Environment:
    def __init__(self, probabilities: List[float]):
        assert len(probabilities) >= 1
        self.probabilities = probabilities

    def n_actions(self) -> int:
        return len(self.probabilities)

    def pull(self, action: int) -> int:
        """Pull the specified bandit (action) and return reward 0 or 1."""
        p = self.probabilities[action]
        return 1 if random.random() < p else 0


class Policy:
    def select_action(self, agent: "Agent") -> int:
        raise NotImplementedError()

    def name(self) -> str:
        return self.__class__.__name__


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.9, decay: float = 0.001, min_epsilon: float = 0.0):
        self.epsilon = float(epsilon)
        self.decay = float(decay)
        self.min_epsilon = float(min_epsilon)

    def select_action(self, agent: "Agent") -> int:
        # Explore
        if random.random() < self.epsilon:
            action = random.randrange(agent.env.n_actions())
        else:
            # Exploit: choose action with highest empirical success rate
            rates = agent.success_rates()
            # break ties randomly
            max_rate = max(rates)
            best = [i for i, r in enumerate(rates) if r == max_rate]
            action = random.choice(best)

        # Apply decay after selection if decay > 0
        if self.decay > 0:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay)

        return action


class ThompsonSamplingPolicy(Policy):
    def select_action(self, agent: "Agent") -> int:
        # For Bernoulli bandits, use Beta(1+wins, 1+losses)
        samples = []
        for a in range(agent.env.n_actions()):
            wins = agent.wins[a]
            pulls = agent.pulls[a]
            losses = pulls - wins
            alpha = 1 + wins
            beta = 1 + max(0, losses)
            samples.append(random.betavariate(alpha, beta))
        # Choose argmax sample (break ties randomly)
        max_s = max(samples)
        best = [i for i, s in enumerate(samples) if s == max_s]
        return random.choice(best)


class Agent:
    def __init__(self, env: Environment, policy: Optional[Policy] = None, memory_size: int = 100):
        self.env = env
        self.policy: Policy = policy or EpsilonGreedyPolicy()
        self.memory_size = int(memory_size)

        # Stats
        self.pulls = [0] * env.n_actions()
        self.wins = [0] * env.n_actions()
        self.cumulative_reward = 0
        self.step_count = 0

        # Memory: list of (action, reward) tuples. If memory_size == 0 -> treat as unlimited.
        self.memory: List[Tuple[int, int]] = []

        # History arrays for plotting
        self.cumulative_reward_history: List[int] = []

    def reset(self):
        self.pulls = [0] * self.env.n_actions()
        self.wins = [0] * self.env.n_actions()
        self.cumulative_reward = 0
        self.step_count = 0
        self.memory.clear()
        self.cumulative_reward_history.clear()

    def set_policy(self, policy: Policy):
        self.policy = policy

    def success_rates(self) -> List[float]:
        rates = []
        for w, p in zip(self.wins, self.pulls):
            rates.append((w / p) if p > 0 else 0.0)
        return rates

    def _push_memory(self, action: int, reward: int):
        self.memory.append((action, reward))
        if self.memory_size > 0:
            # enforce maximum size
            while len(self.memory) > self.memory_size:
                self.memory.pop(0)

    def step(self, action: Optional[int] = None) -> Tuple[int, int]:
        """Perform a single step.

        If `action` is None, the agent selects action via its policy.
        Returns (action, reward).
        """
        if action is None:
            action = self.policy.select_action(self)

        reward = self.env.pull(action)

        # Update stats
        self.pulls[action] += 1
        self.wins[action] += reward
        self.cumulative_reward += reward
        self.step_count += 1

        self._push_memory(action, reward)
        self.cumulative_reward_history.append(self.cumulative_reward)

        return action, reward
"""Core bandit environment, agent, and policy implementations.

Classes:
 - Environment: Bernoulli bandits with fixed probabilities.
 - Agent: interacts with Environment, holds memory and stats.
 - Policies: EpsilonGreedyPolicy and ThompsonSamplingPolicy.

Memory semantics: memory_size=100 (default). If memory_size == 0,
the agent treats it as unlimited (full memory).
"""

import random
from typing import List, Optional, Tuple


class Environment:
    def __init__(self, probabilities: List[float]):
        assert len(probabilities) >= 1
        self.probabilities = probabilities

    def n_actions(self) -> int:
        return len(self.probabilities)

    def pull(self, action: int) -> int:
        """Pull the specified bandit (action) and return reward 0 or 1."""
        p = self.probabilities[action]
        return 1 if random.random() < p else 0


class Policy:
    def select_action(self, agent: "Agent") -> int:
        raise NotImplementedError()

    def name(self) -> str:
        return self.__class__.__name__


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.9, decay: float = 0.001, min_epsilon: float = 0.0):
        self.epsilon = float(epsilon)
        self.decay = float(decay)
        self.min_epsilon = float(min_epsilon)

    def select_action(self, agent: "Agent") -> int:
        # Explore
        if random.random() < self.epsilon:
            action = random.randrange(agent.env.n_actions())
        else:
            # Exploit: choose action with highest empirical success rate
            rates = agent.success_rates()
            # break ties randomly
            max_rate = max(rates)
            best = [i for i, r in enumerate(rates) if r == max_rate]
            action = random.choice(best)

        # Apply decay after selection if decay > 0
        if self.decay > 0:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay)

        return action


class ThompsonSamplingPolicy(Policy):
    def select_action(self, agent: "Agent") -> int:
        # For Bernoulli bandits, use Beta(1+wins, 1+losses)
        samples = []
        for a in range(agent.env.n_actions()):
            wins = agent.wins[a]
            pulls = agent.pulls[a]
            losses = pulls - wins
            alpha = 1 + wins
            beta = 1 + max(0, losses)
            samples.append(random.betavariate(alpha, beta))
        # Choose argmax sample (break ties randomly)
        max_s = max(samples)
        best = [i for i, s in enumerate(samples) if s == max_s]
        return random.choice(best)


class Agent:
    def __init__(self, env: Environment, policy: Optional[Policy] = None, memory_size: int = 100):
        self.env = env
        self.policy: Policy = policy or EpsilonGreedyPolicy()
        self.memory_size = int(memory_size)

        # Stats
        self.pulls = [0] * env.n_actions()
        self.wins = [0] * env.n_actions()
        self.cumulative_reward = 0
        self.step_count = 0

        # Memory: list of (action, reward) tuples. If memory_size == 0 -> treat as unlimited.
        self.memory: List[Tuple[int, int]] = []

        # History arrays for plotting
        self.cumulative_reward_history: List[int] = []

    def reset(self):
        self.pulls = [0] * self.env.n_actions()
        self.wins = [0] * self.env.n_actions()
        self.cumulative_reward = 0
        self.step_count = 0
        self.memory.clear()
        self.cumulative_reward_history.clear()

    def set_policy(self, policy: Policy):
        self.policy = policy

    def success_rates(self) -> List[float]:
        rates = []
        for w, p in zip(self.wins, self.pulls):
            rates.append((w / p) if p > 0 else 0.0)
        return rates

    def _push_memory(self, action: int, reward: int):
        self.memory.append((action, reward))
        if self.memory_size > 0:
            # enforce maximum size
            while len(self.memory) > self.memory_size:
                self.memory.pop(0)

    def step(self, action: Optional[int] = None) -> Tuple[int, int]:
        """Perform a single step.

        If `action` is None, the agent selects action via its policy.
        Returns (action, reward).
        """
        if action is None:
            action = self.policy.select_action(self)

        reward = self.env.pull(action)

        # Update stats
        self.pulls[action] += 1
        self.wins[action] += reward
        self.cumulative_reward += reward
        self.step_count += 1

        self._push_memory(action, reward)
        self.cumulative_reward_history.append(self.cumulative_reward)

        return action, reward
