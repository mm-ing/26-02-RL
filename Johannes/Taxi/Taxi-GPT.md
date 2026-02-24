# Taxi Project Implementation Prompt (Executable)

Use this prompt with GitHub Copilot (GPT-5.3-Codex):

---

You are GitHub Copilot (GPT-5.3-Codex) working in VS Code. Implement a complete Taxi reinforcement learning project in the current directory according to the requirements below.

## Scope
Build a Gymnasium-based Taxi RL app with Tkinter GUI, deep RL agents (PyTorch), trainer, tests, output folders, and README.

Project name: `Taxi`

## Required files to create/update
- `Taxi_app.py`
- `Taxi_logic.py`
- `Taxi_gui.py`
- `tests/test_taxi_logic.py`
- `requirements.txt`
- `README.md`
- create folders `results_csv/` and `plots/` if missing

## Core architecture requirements
Use object-oriented design.

### Environment and logic (`Taxi_logic.py`)
1. Use `gym.make('Taxi-v3')`.
2. Add configurable environment wrapper with settings:
   - `is_raining: bool = False`
   - `fickle_passenger: bool = False`
3. Include sensible, deterministic effect of these toggles on dynamics/rewards while preserving valid Taxi transitions.
4. Implement replay buffer(s), training utilities, and a `Trainer` class with:
   - `run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)`
     - call `progress_callback(step)` every step when provided.
   - `train(policy, num_episodes, max_steps, epsilon, save_csv=None)`
     - optional CSV export of sampled transitions into `results_csv/`.
5. Implement four policy/agent classes with suggested defaults:
   - `DQN`
   - `DoubleDQN`
   - `DuelingDQN`
   - `PrioDQN` (prioritized replay)
6. Use PyTorch and support GPU automatically (`cuda` if available, else CPU).

### GUI (`Taxi_gui.py`)
Use Tkinter + ttk with panel layout and behavior from baseline spec.

#### Layout (must match)
- `Environment` panel (top, full width)
- `Controls` panel (left)
  - Row0: `Reset All` | `Clear Plot`
  - Row1: `Run single episode` | `Save samplings CSV`
  - Row2: `Train and Run` | `Save Plot PNG`
  - buttons `sticky='ew'`
- `Current State` panel below Controls with single-line padded status text:
  - `Training: step:    3  episode:   12`
  - or `    Idle: step:    0  episode:    0`
- `DNN Parameters` panel between Controls and Training Parameters
  - same visual height style as Training Parameters, tight width
- `Training Parameters` panel (tight width, not horizontally expanding)
- `Live Plot` panel (bottom, full width)

#### Environment panel behavior
- Display Taxi visualization/animation (Gymnasium render integration; pygame-based display path where applicable).
- Add checkboxes:
  - `raining` -> binds `is_raining`
  - `running man` -> binds `fickle_passenger`

#### Training parameters
Include at least:
- `max_steps`
- `episodes`
- policy dropdown (`DQN`, `DoubleDQN`, `DuelingDQN`, `PrioDQN`)
- `Live plot` checkbox (default `True`)
- `reduced speed` checkbox (default `True`)
- `MA N values` input for moving average window (default `20`)
- `Render every N steps` input to control Taxi animation refresh rate during training (default `10`)

#### DNN parameters
Expose relevant editable hyperparameters (e.g., learning rate, gamma, batch size, buffer size, target update, epsilon min/max/decay, hidden size).

#### Plot behavior
- Embedded matplotlib canvas.
- Plot episode rewards for all runs.
- Plot moving average curve over `N` values (from `MA N values`) with doubled linewidth.
- Legend outside right.
- Each legend entry gets a toggle checkbox; untoggling hides corresponding line.
- Throttle redraws to ~150 ms using timestamp tracking (`self._last_plot_update`).

#### Threading/responsiveness
- Training runs in background thread.
- All GUI updates scheduled on main thread via `after(0, ...)`.
- Per-step updates modify `Current State` directly (no full canvas redraw).
- During training, refresh the Taxi animation every `N` steps from `Render every N steps` (minimum effective value: 1).
- If `reduced speed` enabled: sleep `0.033` sec between episodes.

#### Controls behavior
- `Run single episode`:
  - reset agent/start state,
  - set `current_episode = 1`, `current_step = 0`,
  - run one episode via trainer and animate sequential transitions.
- `Train and Run`:
  - for each episode, update episode/step via `after(0, ...)`,
  - pass `progress_callback` to `run_episode`,
  - append reward and update live plot when enabled.
- `Reset All`:
  - request stop (`_stop_requested = True`),
  - clear runs/plot/map,
  - reset current agent instance.
- `Save samplings CSV`:
  - call trainer training/export to `results_csv/`.
- `Save Plot PNG`:
  - save current matplotlib figure to `plots/` with filename including:
    - policy, epsilon min/max, alpha, gamma, episodes, max_steps, timestamp.

## App entry (`Taxi_app.py`)
- Wire up environment, selected policy agent, trainer, and GUI.
- Launch Tk main loop.

## Tests (`tests/test_taxi_logic.py`)
Use `pytest` and include tests that cover:
- environment `step`
- `is_reachable` behavior (implement helper if missing in Taxi context)
- `Trainer.run_episode` for multiple policies
- basic learning update signal/weight update path

## Dependencies (`requirements.txt`)
Include runtime/test dependencies:
- `gymnasium`
- `pygame`
- `numpy`
- `torch`
- `matplotlib`
- `pillow`
- `pytest`

Do **not** change the Python environment configuration; only update project files.

## README (`README.md`)
Document:
- architecture
- policy differences
- configurable toggles (`raining`, `running man`)
- GUI usage
- run instructions
- test instructions
- output files (`results_csv/`, `plots/`)

## Quality bar
- Keep code modular and readable.
- Prefer minimal, robust defaults.
- Avoid changing unrelated files.
- Run focused tests for the created test file and fix issues.

## Execution style
1. Inspect current Taxi directory first.
2. Implement files.
3. Run tests.
4. Summarize what changed and any known limitations.

---

If anything is ambiguous, choose the simplest interpretation that satisfies all listed requirements.