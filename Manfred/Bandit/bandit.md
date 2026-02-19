# Multi-armed bandit app — spec / prompt

## Goal
- implement a small experiment app for a 3-armed bandit problem using epsilon-greedy with decay, with both manual and agent-driven interaction via a Tkinter GUI.

## Role
- I am a solftware engineer who want to learn reinforcement learning

## Files to create
- `bandits_app.py` — entrypoint: imports logic and GUI, starts the UI.
- `bandit_logic.py` — core classes and functions:
  - `OpenArmedBandit` class
    - constructor: `__init__(self, reward_prob)` where `reward_prob` is the true success probability (Bernoulli rewards 0/1).
    - method `pull(self) -> int`: return 1 on success, 0 on failure (random Bernoulli).
  - `Agent` class (epsilon-greedy with decay and finite memory)
    - constructor: `__init__(self, n_arms: int, epsilon: float = 0.1, decay: float = 1.0, memory: int = 0)`
      - `memory` = number of last pulls to use to compute empirical mean per arm (0 = use all history).
      - `decay` multiplies epsilon after each agent action (or after each loop iteration).
    - `select_action(self) -> int`: choose an arm index (0..n_arms-1).
    - `update(self, arm: int, reward: int) -> None`: record outcome and update estimates.
    - `run(self, envs: List[OpenArmedBandit], n_loops: int) -> List[int]`: run `n_loops` pulls, return list of rewards or summary stats.
    - Expose simple stats: pulls per arm, successes per arm, cumulative rewards, success rates.
  - `EpsilonGreedyPolicy` class
    - extract epsilon greedy logic into class and let the agent use this class
  - `ThompsonSamplingPolicy` class
    - extract "Thompson Sampling (Bayesian Bandits)" logic into class and let the agent use this class    
  - Keep logic independent of GUI; provide small helper functions to run a single agent step or many steps.
- `bandits_app.py`
  - Imports `OpenArmedBandit` as environment and `Agent` from `bandit_logic`.
  - Initialize three bandits with configurable true probabilities (defaults e.g. [0.2, 0.5, 0.8]).
  - Inject bandits and agent into gui, then start the gui.

## GUI: `bandit_gui.py` (Tkinter)
- Layout:
  - Controls:
    - Entry for "Agent loops (n)" — integer.
    - Entry for "Agent memory (last n pulls)" — integer (0 = all history).
    - Entry or spinboxes for agent epsilon and decay (optional).
    - Three buttons for manual selection: "Pull Bandit 1", "Pull Bandit 2", "Pull Bandit 3".
    - Button "Agent: single step" — agent selects and pulls once.
    - Button "Agent: run n loops" — run configured n loops.
    - Dropdown "Method" to select epsilon greedy or thompson samling and let the agent know, which method to pic.
    - Optional: button "Reset" to restart counts.

  - Displays:
    - Live cumulative reward plot.
      - Use two different colors for each method within plot, to compare both methods
      - Add a legend for methods to plot
    - Summary table/text: total loops, for each bandit show total pulls, total reward (successes), and success rate.
    - (Optional) Small plot or list of cumulative reward over time (matplotlib embed is acceptable).

- Behavior:
  - Manual button: performs a pull on selected bandit, updates counts and GUI.
  - Agent single: uses current Agent instance to select_action(), pulls chosen bandit, calls agent.update(), decays epsilon, update GUI.
  - Agent run n: run agent.run(envs, n_loops) or loop single step n times; after finish update GUI and summary.
  - Show cumulative reward over all pulls (manual + agent) and number of total loops executed.
  - Switching methods means recreate agent and reset values, but let the last plot intact.
  - Ensure GUI is responsive (use after() or run agent loops in short batches if needed).

## Metrics & output
- Cumulative reward over all pulls (displayed prominently).
  - 
- Summary showing:
  - Total loops (pulls).
  - Bandit1: pulls / total reward / success rate.
  - Bandit2: pulls / total reward / success rate.
  - Bandit3: pulls / total reward / success rate.

## Usage
- Run with: `python bandits_app.py`
- bandits_app should create bandit objects, create an Agent with default params, and start the Tkinter GUI.

## Implementation notes
- Use Bernoulli rewards (0/1) so success rate is clear.
- Agent memory: maintain per-arm deque of most recent rewards (size = memory) or full history if memory == 0.
- Epsilon decay: multiply epsilon by `decay` after each agent action (document choice).
- Keep separation between logic and UI: GUI imports and calls methods but does not implement learning logic.

# End.


