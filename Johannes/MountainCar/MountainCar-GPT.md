# MountainCar Implementation Prompt (Executable by GPT-5.3-Codex)

You are GPT-5.3-Codex operating as a coding agent in this workspace. Implement the **MountainCar** RL project exactly in this folder:

- `26-02-RL/Johannes/MountainCar/`

Use and merge requirements from:

- `26-02-RL/Johannes/RL_initial_2.0.md`
- `26-02-RL/Johannes/MountainCar/MountainCar.md`

Do not ask for permission before coding. Execute end-to-end: create files, implement code, run tests, and summarize results.

---

## 1) Files to create

Create these files in `26-02-RL/Johannes/MountainCar/`:

- `MountainCar_app.py`
- `MountainCar_logic.py`
- `MountainCar_gui.py`
- `README.md`
- `requirements.txt`
- `tests/test_MountainCar_logic.py`

Create directories if missing:

- `results_csv/`
- `plots/`
- `tests/`

Do not modify Python environment configuration globally.

---

## 2) Core architecture requirements

Use object-oriented design and PyTorch (GPU-aware if available) for deep RL policies.

### Environment wrapper

Implement a Mountain Car environment wrapper around:

- `gym.make('MountainCar-v0')`

Support configurable environment setup values:

- `goal_velocity` (default `0.0`)
- `x_init` (default `math.pi`)
- `y_init` (default `1.0`)

Notes:
- Gymnasium MountainCar reset supports options; if a field is unsupported by the env API, handle gracefully with fallback behavior and keep app running.
- Provide pygame-based rendering path and return RGB frames compatible with GUI image display.

### Policies to implement

Implement three selectable approaches:

1. **Dueling DQN**
   - class name: `DuelingDQN`
2. **D3QN (Double + Dueling DQN)**
   - class name: `D3QN`
3. **Double DQN + Prioritized Experience Replay**
   - class name: `DDQN_PER` (Python-safe class name; map GUI label `DDQN+PER`)

Provide per-policy internal parameter storage and independent defaults.

### Suggested defaults (implement)

- Shared defaults:
  - `gamma=0.99`
  - `learning_rate=1e-3`
  - `replay_size=50000`
  - `batch_size=64`
  - `target_update=200`
  - `hidden_layer_size=128`
- Exploration defaults:
  - `epsilon_max=1.0`
  - `epsilon_decay=0.995`
  - `epsilon_min=0.05`
- Policy-specific additions:
  - `DDQN_PER`: include PER params such as `alpha=0.6`, `beta_start=0.4`, `beta_frames=50000`, `eps_prio=1e-5`

### Trainer API (must match)

Implement in logic module:

- `run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)`
  - Calls `progress_callback(step)` for each step when provided.
- `train(policy, num_episodes, max_steps, epsilon, save_csv=None)`
  - Trains for multiple episodes.
  - Optionally writes sampled transitions to CSV in `results_csv/`.

Keep behavior deterministic when seeds are provided.

---

## 3) GUI requirements (`tkinter` + embedded matplotlib)

Implement full GUI in `MountainCar_gui.py` with layout and behavior from `RL_initial_2.0.md`.

### Panels/layout

Main window grid with:

- `Environment` panel (left, top)
- `Parameters` panel (right, tight width, scrollable as needed)
- `Controls` panel (full width)
- `Current Run` panel (full width)
- `Live Plot` panel (bottom full width)

### Parameter panel groups

Use 4-column row pattern: `label, input, label, input`.

Groups:

1. **Environment**
   - `Update` button
   - `animation FPS` (default 10)
   - `goal velocity`
   - `x`
   - `y`

2. **General**
   - policy dropdown values:
     - `Dueling DQN`
     - `D3QN`
     - `DDQN+PER`
   - `max steps`, `episodes`, `epsilon max`, `epsilon decay`, `epsilon min`

3. **Specific**
   - show applicable inputs per policy:
     - `gamma`, `learning rate`, `replay size`, `batch size`, `target update`, `hidden layer size`
     - plus PER fields when policy is `DDQN+PER`

4. **Live Plot**
   - `moving average values` (default 20)

### Controls row (equal-width buttons)

Buttons left-to-right:

- `Run single episode`
- `Train and Run`
- `Pause` / `Run` toggle
- `Reset All`
- `Clear Plot`
- `Save samplings CSV`
- `Save Plot PNG`

All buttons use horizontal expansion in cell (`sticky='ew'`).

### Current Run panel

- Steps progress bar + label
- Episodes progress bar + label
- Epsilon line formatted as `Epsilon: 0.90`

### Live Plot panel

- Embedded matplotlib canvas plotting episode rewards.
- Plot current run reward (thin, semi-transparent) and moving average (same color, bold).
- Legend outside right.
- Legend entries are clickable to show/hide line.
- Hidden legend entries appear visually de-emphasized.
- Label entries include policy and relevant parameters.

---

## 4) Threading/performance constraints (must enforce)

- Training runs in background thread.
- Worker thread does **not** read Tk variables directly.
- Snapshot all UI values on main thread before starting worker.
- Worker thread does **not** update widgets directly.
- Use main-thread UI pump every ~20–50 ms to consume latest pending state.
- Coalesce updates (latest episode/step/frame/rewards/finalize flags only).
- Per-step state updates must avoid full canvas redraw.
- Throttle plot redraws to ~150 ms via timestamp (e.g. `self._last_plot_update`).
- `Reset All` and `Clear Plot` must be safe during active training.

---

## 5) User interaction semantics

Implement exactly:

- **Run single episode**
  - reset agent to configured start while preserving learned policy when available
  - run one `Trainer.run_episode(...)`
  - animate transitions sequentially

- **Train and Run**
  - run training in background
  - feed `progress_callback` into `run_episode`
  - update pending state for UI pump
  - append per-episode reward for live plotting

- **Pause/Run**
  - pause and resume training loop safely

- **Reset All**
  - set stop request flag
  - clear plot runs and environment view
  - reset agent instance safely during/after training

- **Clear Plot**
  - remove plotted lines without crashing

- **Save samplings CSV**
  - write sampled transitions into `results_csv/`

- **Save Plot PNG**
  - save figure into `plots/`
  - filename must encode: policy, eps min/max, learning rate, gamma, episodes, max_steps, timestamp

---

## 6) Tests and validation

Create `tests/test_MountainCar_logic.py` with pytest tests for:

- environment `step`
- `is_reachable`
- `Trainer.run_episode` under multiple policies
- basic learning update behavior

Add GUI smoke tests as lightweight checks (where feasible) for:

- responsiveness during longer run with live plot enabled
- repeated `Clear Plot` / `Reset All` during training without exceptions
- interactive legend toggling after multiple runs and clear/reset cycles

Run tests and report results.

---

## 7) Dependencies

Populate `requirements.txt` with at least:

- `gymnasium`
- `pygame`
- `numpy`
- `torch`
- `matplotlib`
- `pillow`
- `pytest`

Keep versions reasonably compatible with Python 3.8+.

---

## 8) README

Create `README.md` with:

- project overview
- policy descriptions
- file structure
- run instructions
- test instructions
- notes on GPU usage and fallback
- output locations (`results_csv/`, `plots/`)

---

## 9) Implementation quality bar

- Keep changes minimal, focused, and consistent style.
- Avoid unrelated refactors.
- Handle edge cases and invalid parameter input gracefully.
- Ensure imports and paths are Windows-safe.
- Ensure app starts from `MountainCar_app.py`.

---

## 10) Deliverable summary format (after coding)

After implementation and validation, provide:

1. What changed (concise)
2. Files created/updated
3. Test command(s) + outcomes
4. Any known limitations
5. Next optional improvements
