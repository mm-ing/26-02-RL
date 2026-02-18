from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


class Bandit:
    def __init__(
        self,
        name: str,
        start_coins: int,
        rng: Optional[random.Random] = None,
        probability_mode: str = "variable",
    ) -> None:
        self.name = name
        self.initial_coins = int(start_coins)
        self.rng = rng or random.Random()
        self.probability_mode = probability_mode
        self.constant_probability = min(self.initial_coins / 100.0, 1.0)
        self.reset()

    def reset(self, start_coins: Optional[int] = None) -> None:
        if start_coins is not None:
            self.initial_coins = int(start_coins)
        self.constant_probability = min(self.initial_coins / 100.0, 1.0)
        self.stored_coins = int(self.initial_coins)
        self.pulls = 0
        self.success = 0
        self.total_returned = 0

    def set_probability_mode(self, probability_mode: str) -> None:
        if probability_mode not in {"constant", "variable"}:
            raise ValueError("Unbekannter probability_mode. Erlaubt: 'constant', 'variable'.")
        self.probability_mode = probability_mode

    @property
    def success_rate(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.success / self.pulls

    def pull(self) -> int:
        self.pulls += 1
        self.stored_coins += 1

        if self.probability_mode == "constant":
            payout_probability = self.constant_probability
        else:
            payout_probability = min(self.stored_coins / 100.0, 1.0)
        payout = 0
        if self.rng.random() < payout_probability:
            payout = self.rng.randint(1, self.stored_coins)
            self.success += 1

        self.stored_coins -= payout
        reward = 1 if payout > 0 else 0
        self.total_returned += reward
        return reward

    def get_state(self) -> Dict[str, float]:
        return {
            "bandit": self.name,
            "pulls": self.pulls,
            "success": self.success,
            "success_rate": self.success_rate,
            "cumulative_reward": self.total_returned,
        }


class Environment:
    def __init__(
        self,
        start_amounts: Tuple[int, int, int] = (20, 40, 80),
        seed: Optional[int] = None,
        probability_mode: str = "variable",
    ) -> None:
        if len(start_amounts) != 3:
            raise ValueError("Es müssen genau drei Startmengen übergeben werden.")
        if probability_mode not in {"constant", "variable"}:
            raise ValueError("Unbekannter probability_mode. Erlaubt: 'constant', 'variable'.")

        self.start_amounts = tuple(int(value) for value in start_amounts)
        self.seed = seed
        self.probability_mode = probability_mode
        self._base_rng = random.Random(seed)
        self.bandits = [
            Bandit(
                name=f"Bandit {index + 1}",
                start_coins=amount,
                rng=random.Random(self._base_rng.random()),
                probability_mode=self.probability_mode,
            )
            for index, amount in enumerate(self.start_amounts)
        ]

    def set_probability_mode(self, probability_mode: str) -> None:
        if probability_mode not in {"constant", "variable"}:
            raise ValueError("Unbekannter probability_mode. Erlaubt: 'constant', 'variable'.")
        self.probability_mode = probability_mode
        for bandit in self.bandits:
            bandit.set_probability_mode(probability_mode)

    def step(self, action: int) -> int:
        if action < 0 or action >= len(self.bandits):
            raise IndexError("Action außerhalb des gültigen Bandit-Bereichs.")
        return self.bandits[action].pull()

    def reset(self) -> None:
        for index, bandit in enumerate(self.bandits):
            bandit.reset(self.start_amounts[index])

    def get_state(self) -> List[Dict[str, float]]:
        return [bandit.get_state() for bandit in self.bandits]


@dataclass
class Policy:
    name: str

    def choose_action(self, agent: "Agent", env: Environment) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: int) -> None:
        return

    def reset(self, n_bandits: int) -> None:
        return


class EpsilonGreedy(Policy):
    def __init__(self, epsilon_start: float = 0.9, epsilon_decay: float = 0.01, epsilon_min: float = 0.01) -> None:
        super().__init__(name="Epsilon-Greedy")
        self.epsilon_start = float(epsilon_start)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.epsilon = float(epsilon_start)

    def choose_action(self, agent: "Agent", env: Environment) -> int:
        if random.random() < self.epsilon:
            return random.randrange(len(env.bandits))

        best_value = max(agent.q_values)
        best_actions = [index for index, value in enumerate(agent.q_values) if value == best_value]
        return random.choice(best_actions)

    def update(self, action: int, reward: int) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * (1.0 - self.epsilon_decay))

    def reset(self, n_bandits: int) -> None:
        self.epsilon = float(self.epsilon_start)

    def configure(self, epsilon_start: Optional[float] = None, epsilon_decay: Optional[float] = None) -> None:
        if epsilon_start is not None:
            self.epsilon_start = float(epsilon_start)
            self.epsilon = float(epsilon_start)
        if epsilon_decay is not None:
            self.epsilon_decay = float(epsilon_decay)


class ThompsonSampling(Policy):
    def __init__(self) -> None:
        super().__init__(name="Thompson Sampling")
        self.alpha: np.ndarray = np.array([])
        self.beta: np.ndarray = np.array([])

    def choose_action(self, agent: "Agent", env: Environment) -> int:
        if self.alpha.size != len(env.bandits):
            self.reset(len(env.bandits))

        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: int) -> None:
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1

    def reset(self, n_bandits: int) -> None:
        self.alpha = np.ones(n_bandits, dtype=float)
        self.beta = np.ones(n_bandits, dtype=float)


class Agent:
    def __init__(
        self,
        environment: Environment,
        loops: int = 100,
        memory: int = 0,
        epsilon_start: float = 0.9,
        epsilon_decay: float = 0.01,
    ) -> None:
        self.environment = environment
        self.default_loops = int(loops)
        self.memory = int(memory)

        self.policies: Dict[str, Policy] = {
            "Epsilon-Greedy": EpsilonGreedy(epsilon_start=epsilon_start, epsilon_decay=epsilon_decay),
            "Thompson Sampling": ThompsonSampling(),
        }
        for policy in self.policies.values():
            policy.reset(len(self.environment.bandits))

        self.active_policy_name = "Epsilon-Greedy"
        self.reset_learning()

    @property
    def active_policy(self) -> Policy:
        return self.policies[self.active_policy_name]

    def set_policy(self, policy_name: str) -> None:
        if policy_name not in self.policies:
            raise ValueError(f"Unbekannte Policy: {policy_name}")
        self.active_policy_name = policy_name

    def configure(self, memory: Optional[int] = None, epsilon: Optional[float] = None, epsilon_decay: Optional[float] = None) -> None:
        if memory is not None:
            self.memory = max(0, int(memory))

        epsilon_policy = self.policies.get("Epsilon-Greedy")
        if isinstance(epsilon_policy, EpsilonGreedy):
            epsilon_policy.configure(epsilon_start=epsilon, epsilon_decay=epsilon_decay)

    def reset_learning(self) -> None:
        n_bandits = len(self.environment.bandits)
        self.action_counts = [0 for _ in range(n_bandits)]
        self.q_values = [0.0 for _ in range(n_bandits)]
        self.actions: List[int] = []
        self.rewards: List[int] = []
        self.total_steps = 0
        self.total_reward = 0
        self.policy_history: Dict[str, List[Tuple[int, int]]] = {name: [] for name in self.policies.keys()}

        for policy in self.policies.values():
            policy.reset(n_bandits)

    def reset(self) -> None:
        self.environment.reset()
        self.reset_learning()

    def _remember(self, action: int, reward: int) -> None:
        self.actions.append(action)
        self.rewards.append(reward)

        if self.memory > 0:
            self.actions = self.actions[-self.memory :]
            self.rewards = self.rewards[-self.memory :]

    def step(self) -> Tuple[int, int]:
        action = self.active_policy.choose_action(self, self.environment)
        reward = self.environment.step(action)

        self.total_steps += 1
        self.total_reward += reward
        self._remember(action, reward)

        self.action_counts[action] += 1
        count = self.action_counts[action]
        current_value = self.q_values[action]
        self.q_values[action] = current_value + (reward - current_value) / count

        self.active_policy.update(action, reward)
        self.policy_history[self.active_policy_name].append((self.total_steps, self.total_reward))

        return action, reward

    def run(self, loops: Optional[int] = None) -> List[Tuple[int, int]]:
        n = self.default_loops if loops is None else int(loops)
        result: List[Tuple[int, int]] = []
        for _ in range(max(0, n)):
            result.append(self.step())
        return result

    def get_bandit_state(self) -> List[Dict[str, float]]:
        return self.environment.get_state()

    def get_best_bandit(self) -> Dict[str, float]:
        states = self.get_bandit_state()
        best = max(states, key=lambda item: item["cumulative_reward"])
        return best
