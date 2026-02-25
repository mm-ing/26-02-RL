# FrozenLake implementation prompt (for GitHub Copilot coding agent)

You are GitHub Copilot (GPT-5.3-Codex). Implement a complete FrozenLake reinforcement-learning project in this folder:

- `26-02-RL/Johannes/FrozenLake/`

Use object-oriented design and follow the architecture/GUI behavior from `../RL_initial.md`, combined with the FrozenLake-specific requirements from `FrozenLake.md`.

## Precedence and scope rules

- Work only inside `Johannes/FrozenLake/`.
- If `RL_initial.md` contains internal inconsistencies, prioritize the **current GUI layout section** (Controls with `Reset All | Clear Plot`) for button layout and control set.
- For project-specific behavior (FrozenLake environment options, DQN variants, DNN panel, moving-average plot), prioritize `FrozenLake.md`.

## Scope and target files

Create or update the following files in `Johannes/FrozenLake/`:

- `frozenlake_app.py`
- `frozenlake_logic.py`
- `frozenlake_gui.py`
- `tests/test_frozenlake_logic.py`
- `requirements.txt`
- `README.md`
- output folders: `results_csv/` and `plots/`

## Core requirements

### 1) Environment and wrappers

Implement a Frozen Lake environment based on Gymnasium:

- Base env creation with `gym.make('FrozenLake-v1', ...)`
- Configurable parameters from GUI:
  - `is_slippery` (default `True`)
  - `map_name` with options `"4x4"` (default) and `"8x8"`
  - `success_rate` (default `1.0/3.0`)

Important for `success_rate`:

- Gym’s default FrozenLake does not directly expose a `success_rate` argument.
- Implement this via a custom transition matrix wrapper or a custom env setup (using `desc`/`P`) so the intended movement probability is `success_rate` and the remaining probability mass is distributed over side directions.
- Keep behavior compatible with both `4x4` and `8x8` maps.

Use pygame-compatible rendering path for animation in the GUI (rgb frame updates are acceptable if direct pygame surface embedding is impractical).

### 2) Policies / agents (PyTorch, GPU-aware)

Implement four deep-RL policy classes in `frozenlake_logic.py`:

- `DQN` (vanilla)
- `DoubleDQN`
- `DuelingDQN`
- `PrioDQN` (prioritized replay)

Requirements:

- Use PyTorch; auto-select device (`cuda` if available, else `cpu`).
- Define practical default hyperparameters and expose key ones in GUI.
- Handle discrete state representation (FrozenLake states) robustly (e.g., one-hot encoding).
- Use target-network syncing where appropriate.
- For `PrioDQN`, implement a minimal prioritized replay buffer with importance-sampling weights.

### 3) Trainer

Provide a `Trainer` class with at least:

- `run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)`
- `train(policy, num_episodes, max_steps, epsilon, save_csv=None)`

Behavior:

- `progress_callback(step)` called each step when provided.
- `train(...)` optionally writes sampled transitions to CSV in `results_csv/`.
- Return per-episode rewards for plotting.

### 4) GUI layout and behavior (Tkinter + matplotlib)

Build a full GUI in `frozenlake_gui.py` based on `RL_initial.md`, with these panels:

- `Environment` (top, full width)
- `Controls` (left)
- `Current State` (below Controls)
- `DNN Parameters` (between Controls and Training Parameters; tight width; same height style as Training Parameters)
- `Training Parameters` (right, tight width)
- `Live Plot` (bottom, full width)

Controls panel buttons (2x3 grid):

- Row 0: `Reset All` | `Clear Plot`
- Row 1: `Run single episode` | `Save samplings CSV`
- Row 2: `Train and Run` | `Save Plot PNG`

`Current State` label format must match padding rule from `RL_initial.md`.

Training/responsiveness constraints:

- Training on background thread.
- GUI updates via `after(0, ...)`.
- Per-step counter updates must avoid full redraw.
- Throttle plot redraw to ~150 ms (`self._last_plot_update`).
- Support `reduced speed` toggle (default `True`) with ~33 ms sleep between episodes.

Environment panel must include:

- FrozenLake animation
- toggle `slippery` (binds to `is_slippery`)
- input `slippery rate` (binds to `success_rate`)
- dropdown `map size` (`4x4`, `8x8` -> `map_name`)

Parameter inputs:

- Training Parameters: include at least episodes, max steps, epsilon min/max, gamma, alpha/learning-rate, policy selector, live plot checkbox, reduced speed checkbox, `MA N values`.
- DNN Parameters: include architecture/training parameters relevant to DQN variants (e.g., hidden size(s), batch size, replay size, target update interval, warmup steps).

### 5) Live plot requirements

In the reward plot:

- Plot episode reward curve.
- Add moving average curve over `N` values (default 20).
- Moving average line must have double line width compared to the base reward line.
- Add editable input `MA N values` in Training Parameters to control window length.
- Keep clickable legend behavior from `RL_initial.md` (toggle run visibility).

### 6) Save/reset interactions

Implement behavior from `RL_initial.md`:

- `Run single episode`: run one full episode with sequential animation.
- `Train and Run`: threaded training with per-step and per-episode UI updates.
- `Reset All`: request stop, clear plots/map, reset policy instance.
- `Clear Plot`: clear only plotted run data while keeping current environment configuration.
- `Save samplings CSV`: write transitions to `results_csv/`.
- `Save Plot PNG`: save figure to `plots/` with parameter-rich filename + timestamp.

## Tests

Create `tests/test_frozenlake_logic.py` with `pytest` tests for:

- environment step behavior and reachability helper(s)
- trainer `run_episode(...)`
- basic learning update path for each policy family (or representative coverage if runtime-heavy)
- replay/prioritized replay smoke behavior

Tests should be deterministic where practical (seeded runs) and reasonably fast.

## Dependencies

Update `requirements.txt` to include runtime/test dependencies used by implementation, minimally:

- `gymnasium`
- `pygame`
- `torch`
- `numpy`
- `matplotlib`
- `pillow`
- `pytest`

(Keep compatibility with current Python environment; do not change interpreter/environment manager.)

## README

Write/update `README.md` with:

- project overview
- implemented algorithms
- file structure
- install and run commands
- test command
- known limitations/performance notes

## Execution and validation steps

After implementing:

1. Run targeted tests for this folder first (`tests/test_frozenlake_logic.py`).
2. Fix only issues related to this task.
3. Ensure no unnecessary unrelated refactors.

## Quality constraints

- Keep code modular and readable.
- Avoid adding unrelated features.
- Keep defaults sensible and documented.
- Match naming exactly for required classes:
  - `DQN`, `DoubleDQN`, `DuelingDQN`, `PrioDQN`

Deliver complete working code in the specified files.