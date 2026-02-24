from __future__ import annotations

import csv
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

State = Tuple[int, int]
Transition = Tuple[State, int, State, int, bool]


class GridMap:
    """Configurable deterministic grid map with blocked cells and reachability checks."""

    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (0, -1),  # up
        1: (0, 1),   # down
        2: (-1, 0),  # left
        3: (1, 0),   # right
    }

    def __init__(
        self,
        rows: int = 3,
        cols: int = 5,
        blocked: Optional[Iterable[State]] = None,
        start: State = (0, 2),
        target: Optional[State] = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.blocked: Set[State] = set(blocked or {(2, 0), (2, 1)})
        self.start = start
        self.target = target if target is not None else (cols - 1, rows - 1)
        self.validate_configuration()

    def in_bounds(self, state: State) -> bool:
        x, y = state
        return 0 <= x < self.cols and 0 <= y < self.rows

    def is_blocked(self, state: State) -> bool:
        return state in self.blocked

    def is_valid_cell(self, state: State) -> bool:
        return self.in_bounds(state) and not self.is_blocked(state)

    def next_state(self, state: State, action: int) -> State:
        dx, dy = self.ACTIONS[action]
        candidate = (state[0] + dx, state[1] + dy)
        if not self.in_bounds(candidate) or self.is_blocked(candidate):
            # Deterministic wall collision behavior: stay in place.
            return state
        return candidate

    def reward(self, next_state: State) -> int:
        return 0 if next_state == self.target else -1

    def is_terminal(self, state: State) -> bool:
        return state == self.target

    def update_dimensions(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.blocked = {cell for cell in self.blocked if self.in_bounds(cell)}
        if not self.in_bounds(self.start):
            self.start = (0, max(0, rows - 1))
        if not self.in_bounds(self.target):
            self.target = (cols - 1, rows - 1)
        self.validate_configuration()

    def set_start(self, state: State) -> None:
        if not self.in_bounds(state):
            raise ValueError("Start must be inside the grid.")
        if self.is_blocked(state):
            raise ValueError("Start cannot be on a blocked cell.")
        if state == self.target:
            raise ValueError("Start and target must be different.")
        old = self.start
        self.start = state
        if not self.has_valid_path():
            self.start = old
            raise ValueError("Start position must keep at least one valid path to target.")

    def set_target(self, state: State) -> None:
        if not self.in_bounds(state):
            raise ValueError("Target must be inside the grid.")
        if self.is_blocked(state):
            raise ValueError("Target cannot be on a blocked cell.")
        if state == self.start:
            raise ValueError("Target and start must be different.")
        old = self.target
        self.target = state
        if not self.has_valid_path():
            self.target = old
            raise ValueError("Target position must keep at least one valid path from start.")

    def toggle_blocked(self, state: State) -> None:
        if not self.in_bounds(state):
            return
        if state in {self.start, self.target}:
            return
        if state in self.blocked:
            self.blocked.remove(state)
            return

        self.blocked.add(state)
        if not self.has_valid_path():
            self.blocked.remove(state)
            raise ValueError("Blocking this cell would remove all valid paths to the target.")

    def set_blocked(self, blocked_cells: Iterable[State]) -> None:
        cleaned: Set[State] = set()
        for cell in blocked_cells:
            if not self.in_bounds(cell):
                raise ValueError(f"Blocked cell {cell} is outside the grid.")
            if cell in {self.start, self.target}:
                raise ValueError("Blocked cells cannot include start or target.")
            cleaned.add(cell)

        old = self.blocked
        self.blocked = cleaned
        if not self.has_valid_path():
            self.blocked = old
            raise ValueError("Blocked configuration must keep at least one valid path.")

    def neighbors(self, state: State) -> List[State]:
        result: List[State] = []
        for action in self.ACTIONS:
            ns = self.next_state(state, action)
            if ns != state:
                result.append(ns)
        return result

    def has_valid_path(self) -> bool:
        if not self.is_valid_cell(self.start) or not self.is_valid_cell(self.target):
            return False
        queue: deque[State] = deque([self.start])
        visited: Set[State] = {self.start}
        while queue:
            current = queue.popleft()
            if current == self.target:
                return True
            for nxt in self.neighbors(current):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def validate_configuration(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Grid size must be positive.")
        if not self.in_bounds(self.start):
            raise ValueError("Start position must be inside the grid.")
        if not self.in_bounds(self.target):
            raise ValueError("Target position must be inside the grid.")
        if self.start == self.target:
            raise ValueError("Start and target cannot be identical.")
        for cell in self.blocked:
            if not self.in_bounds(cell):
                raise ValueError(f"Blocked cell {cell} must be inside the grid.")
            if cell in {self.start, self.target}:
                raise ValueError("Blocked cells cannot include start or target.")
        if not self.has_valid_path():
            raise ValueError("Configuration must preserve at least one valid path.")


class Agent:
    """Agent state holder to keep environment transitions explicit and testable."""

    def __init__(self, start: State) -> None:
        self.start = start
        self.position = start

    def reset(self, start: Optional[State] = None) -> State:
        if start is not None:
            self.start = start
        self.position = self.start
        return self.position


class Environment:
    """Deterministic environment implementing p(s, a) -> s' and reward model."""

    def __init__(self, grid_map: GridMap) -> None:
        self.grid_map = grid_map
        self.agent = Agent(grid_map.start)

    def reset(self) -> State:
        self.agent.start = self.grid_map.start
        return self.agent.reset()

    def transition(self, state: State, action: int) -> Transition:
        next_state = self.grid_map.next_state(state, action)
        reward = self.grid_map.reward(next_state)
        done = self.grid_map.is_terminal(next_state)
        return state, action, next_state, reward, done

    def step(self, action: int) -> Transition:
        transition = self.transition(self.agent.position, action)
        self.agent.position = transition[2]
        return transition


@dataclass
class EpisodeTrajectory:
    transitions: List[Transition]

    def __init__(self) -> None:
        self.transitions = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()


class EpsilonScheduler:
    """Exponential epsilon decay from max to min over episodes."""

    def __init__(self, epsilon_max: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995) -> None:
        if not (0.0 <= epsilon_min <= epsilon_max <= 1.0):
            raise ValueError("Epsilon bounds must satisfy 0 <= min <= max <= 1.")
        if not (0.0 < epsilon_decay <= 1.0):
            raise ValueError("Epsilon decay must be in (0, 1].")
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def value(self, episode_index: int) -> float:
        value = self.epsilon_max * (self.epsilon_decay ** max(0, episode_index))
        return max(self.epsilon_min, value)


class MonteCarlo:
    """First-visit Monte Carlo control using state-value estimation V(s)."""

    def __init__(self, gamma: float = 0.99) -> None:
        self.gamma = gamma
        self.value_table: DefaultDict[State, float] = defaultdict(float)
        self.returns_sum: DefaultDict[State, float] = defaultdict(float)
        self.returns_count: DefaultDict[State, int] = defaultdict(int)

    def select_action(self, grid_map: GridMap, state: State, epsilon: float) -> int:
        if random.random() < epsilon:
            # Epsilon-greedy: with probability epsilon, explore random actions.
            return random.choice(list(GridMap.ACTIONS.keys()))

        action_values: Dict[int, float] = {}
        for action in GridMap.ACTIONS:
            next_state = grid_map.next_state(state, action)
            action_values[action] = self.value_table[next_state]
        best_value = max(action_values.values())
        candidates = [action for action, value in action_values.items() if value == best_value]
        return random.choice(candidates)

    def finish_episode(self, trajectory: Sequence[Transition]) -> None:
        visited: Set[State] = set()
        discounted_return = 0.0

        # Monte Carlo return: compute G_t backwards so each state gets its future discounted return.
        for prev_state, _action, _next_state, reward, _done in reversed(trajectory):
            discounted_return = reward + self.gamma * discounted_return
            if prev_state in visited:
                continue
            visited.add(prev_state)
            self.returns_sum[prev_state] += discounted_return
            self.returns_count[prev_state] += 1
            self.value_table[prev_state] = self.returns_sum[prev_state] / self.returns_count[prev_state]


class QLearning:
    """Tabular Q-learning with epsilon-greedy behavior policy."""

    def __init__(self, alpha: float = 0.2, gamma: float = 0.99) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: DefaultDict[Tuple[State, int], float] = defaultdict(float)

    def get_q(self, state: State, action: int) -> float:
        return self.q_table[(state, action)]

    def best_action(self, state: State) -> int:
        values = {action: self.get_q(state, action) for action in GridMap.ACTIONS}
        best_value = max(values.values())
        candidates = [action for action, value in values.items() if value == best_value]
        return random.choice(candidates)

    def select_action(self, state: State, epsilon: float) -> int:
        if random.random() < epsilon:
            # Epsilon-greedy exploration keeps policy from getting stuck in local choices.
            return random.choice(list(GridMap.ACTIONS.keys()))
        return self.best_action(state)

    def update(self, prev_state: State, action: int, reward: int, next_state: State) -> None:
        current_q = self.get_q(prev_state, action)
        max_next_q = max(self.get_q(next_state, next_action) for next_action in GridMap.ACTIONS)
        td_target = reward + self.gamma * max_next_q

        # Q-learning update applies a one-step TD target with learning rate alpha.
        self.q_table[(prev_state, action)] = current_q + self.alpha * (td_target - current_q)


class SamplingHandler:
    """Stores trajectory samples and exports them as CSV rows."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, object]] = []

    def add_transition(
        self,
        *,
        policy_name: str,
        episode: int,
        step: int,
        transition: Transition,
    ) -> None:
        prev_state, action, next_state, reward, done = transition
        self.rows.append(
            {
                "policy": policy_name,
                "episode": episode,
                "step": step,
                "prev_x": prev_state[0],
                "prev_y": prev_state[1],
                "action": action,
                "next_x": next_state[0],
                "next_y": next_state[1],
                "reward": reward,
                "done": int(done),
            }
        )

    def clear(self) -> None:
        self.rows.clear()

    def export_csv(
        self,
        *,
        output_dir: Path,
        policy_name: str,
        alpha: float,
        gamma: float,
        episodes: int,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_policy = policy_name.replace(" ", "_")
        file_name = f"samplings_{safe_policy}_alpha{alpha}_gamma{gamma}_episodes{episodes}_{timestamp}.csv"
        path = output_dir / file_name

        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "policy",
                    "episode",
                    "step",
                    "prev_x",
                    "prev_y",
                    "action",
                    "next_x",
                    "next_y",
                    "reward",
                    "done",
                ],
            )
            writer.writeheader()
            writer.writerows(self.rows)

        return path


class GridWorldLab:
    """Main coordination class exposing step-wise APIs for GUI animation."""

    POLICY_Q_LEARNING = "Q-learning"
    POLICY_MONTE_CARLO = "Monte Carlo"

    def __init__(self, grid_map: Optional[GridMap] = None) -> None:
        self.grid_map = grid_map if grid_map is not None else GridMap()
        self.environment = Environment(self.grid_map)
        self.q_learning = QLearning(alpha=0.2, gamma=0.99)
        self.monte_carlo = MonteCarlo(gamma=0.99)
        self.scheduler = EpsilonScheduler()
        self.sampling = SamplingHandler()

        self.policy_name = self.POLICY_Q_LEARNING
        self.max_steps_per_episode = 100

        self.current_episode_index = 0
        self.current_step_index = 0
        self.episode_reward = 0
        self.trajectory = EpisodeTrajectory()
        self.last_transition: Optional[Transition] = None

    def set_policy(self, policy_name: str) -> None:
        if policy_name not in {self.POLICY_Q_LEARNING, self.POLICY_MONTE_CARLO}:
            raise ValueError("Unsupported policy.")
        self.policy_name = policy_name

    def configure_learning(
        self,
        *,
        gamma: float,
        alpha: float,
        max_steps_per_episode: int,
        epsilon_max: float,
        epsilon_min: float,
        epsilon_decay: float,
    ) -> None:
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("Gamma must be in [0, 1].")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Alpha must be in (0, 1].")
        if max_steps_per_episode <= 0:
            raise ValueError("Max steps per episode must be positive.")

        self.max_steps_per_episode = max_steps_per_episode
        self.scheduler = EpsilonScheduler(epsilon_max, epsilon_min, epsilon_decay)
        self.q_learning.alpha = alpha
        self.q_learning.gamma = gamma
        self.monte_carlo.gamma = gamma

    def reset_episode(self) -> State:
        self.current_step_index = 0
        self.episode_reward = 0
        self.trajectory.clear()
        return self.environment.reset()

    def _select_action(self, state: State) -> int:
        epsilon = self.scheduler.value(self.current_episode_index)
        if self.policy_name == self.POLICY_Q_LEARNING:
            return self.q_learning.select_action(state, epsilon)
        return self.monte_carlo.select_action(self.grid_map, state, epsilon)

    def step(self) -> Transition:
        state = self.environment.agent.position
        action = self._select_action(state)
        transition = self.environment.step(action)
        prev_state, chosen_action, next_state, reward, done = transition

        self.current_step_index += 1
        self.episode_reward += reward
        self.last_transition = transition
        self.trajectory.add(transition)

        if self.policy_name == self.POLICY_Q_LEARNING:
            # Q-learning updates online after every transition.
            self.q_learning.update(prev_state, chosen_action, reward, next_state)

        self.sampling.add_transition(
            policy_name=self.policy_name,
            episode=self.current_episode_index,
            step=self.current_step_index,
            transition=transition,
        )
        return transition

    def finish_episode(self) -> None:
        if self.policy_name == self.POLICY_MONTE_CARLO and self.trajectory.transitions:
            self.monte_carlo.finish_episode(self.trajectory.transitions)

    def run_episode(self) -> int:
        self.reset_episode()
        for _ in range(self.max_steps_per_episode):
            _prev, _action, _next, _reward, done = self.step()
            if done:
                break
        self.finish_episode()
        total_reward = self.episode_reward
        self.current_episode_index += 1
        return total_reward

    def apply_grid_configuration(
        self,
        *,
        rows: int,
        cols: int,
        blocked: Iterable[State],
        start: State,
        target: State,
    ) -> None:
        self.grid_map.update_dimensions(rows, cols)
        self.grid_map.set_start(start)
        self.grid_map.set_target(target)
        self.grid_map.set_blocked(blocked)
        self.environment.reset()

    def current_state(self) -> State:
        return self.environment.agent.position

    def get_value_table(self) -> Dict[State, float]:
        return dict(self.monte_carlo.value_table)

    def get_q_table(self) -> Dict[Tuple[State, int], float]:
        return dict(self.q_learning.q_table)

    def ensure_valid_path(self) -> bool:
        return self.grid_map.has_valid_path()

    def export_samplings_csv(self, output_dir: Path, episodes: int) -> Path:
        return self.sampling.export_csv(
            output_dir=output_dir,
            policy_name=self.policy_name,
            alpha=self.q_learning.alpha,
            gamma=self.q_learning.gamma,
            episodes=episodes,
        )
