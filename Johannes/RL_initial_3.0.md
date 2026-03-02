# RL Master Blueprint (v2.0)

## Purpose
This file is the **base specification** used together with a second, short project-specific file.

The second file should contain only:
1. `project_name`
2. environment details
3. policy list to expose
4. explicit statement to use `Stable-Baselines3 (SB3)`

With those two files together, the project should be reproducible.

---

## Two-file contract

### File A (this file): fixed shared architecture
Defines:
- folder/file layout
- GUI layout and behavior
- trainer/threading behavior
- plotting/export/testing standards
- naming/styling conventions

### File B (project-specific, minimal)
Must define:
- `project_name`
- environment specification (Gym/Gymnasium ID, environment kwargs, continuous/discrete mode rules)
- allowed policies (display names in GUI)
- backend requirement: `SB3`

Optional in File B:
- project-specific default values (episodes, max_steps, etc.)
- additional environment controls

If File B omits optional values, use defaults from this master file.

---

## Technology and backend requirements
- Language: Python 3.8+
- GUI: Tkinter + ttk
- Plotting: matplotlib (TkAgg)
- Image rendering: Pillow (if available)
- RL backend: **Stable-Baselines3 / sb3-contrib only**
- Deep learning runtime: PyTorch (as used by SB3)
- Environment API: gymnasium

Do not implement custom non-SB3 training algorithms when File B requests SB3.

---

## Project output layout
Create (using `project_name` from File B):
- `<project_name>_app.py`
- `<project_name>_logic.py`
- `<project_name>_gui.py`
- `tests/test_<project_name>_logic.py`
- `tests/test_<project_name>_gui.py`
- `requirements.txt`
- `README.md`
- output folders: `results_csv/`, `plots/`

---

## Core architecture

### Logic module
Implement:
- environment wrapper class
- SB3 policy agent wrapper
- trainer class

Recommended separation:
- `Algorithm` layer: model construction, action selection, update primitives
- `TrainLoop` layer: episode/step orchestration, stop/pause handling, callback/event emission
- keep UI-independent training orchestration classes reusable for headless runs/tests

Trainer capabilities:
- `run_episode(...)`
- `train(...)`
- `evaluate_policy(...)`
- environment rebuild/update
- optional CSV export of sampled transitions
- periodic deterministic evaluation checkpoints

### GUI module
Implement one main GUI class with:
- environment rendering panel
- parameters panel (scrollable)
- controls row
- current-run status/progress
- live matplotlib plot
- background training threads with main-thread UI pump

Preferred threading bridge:
- use a queue/event-bridge between worker threads and GUI (`after()`-driven polling)
- publish structured events (step/episode/training-done/error) rather than direct widget calls

### App module
- simple entrypoint that creates `Tk()`, instantiates GUI, starts `mainloop()`.

---

## SB3 policy behavior
- Expose exactly the policy list from File B.
- Keep GUI display names even if internal mapping requires valid Python identifiers.
- Maintain per-policy default config snapshots.
- Reinitialize policy weights for each new `Train and Run` launch.
- Keep deterministic evaluation separate from exploration training episodes.

Typical mapping (if used):
- DQN-like labels -> SB3 `DQN`
- PPO -> SB3 `PPO`
- A2C -> SB3 `A2C`
- SAC -> SB3 `SAC`
- TRPO -> `sb3_contrib.TRPO` (fallback strategy only if explicitly defined)

---

## GUI layout (must match)
Top-level grid:
- row 0: `Environment` (left), `Parameters` (right)
- row 1: `Controls` (full width)
- row 2: `Current Run` (full width)
- row 3: `Live Plot` (full width)

Top row column ratio:
- Environment : Parameters = **2 : 1**

### Environment panel
- dedicated render canvas
- keep aspect ratio, center image
- redraw on resize
- render in main thread only

### Parameters panel
- scrollable content region
- groups in this order:
  1. `Environment`
  2. `Compare`
  3. `General`
  4. `Specific`
  5. `Live Plot`

- mouse wheel active only while cursor is inside panel
- show vertical scrollbar only if needed

#### Environment group fields
- `Animation on` (default: `True`)
- `Animation FPS` (default: `10`)
- `Update rate (episodes)` (default: `5`)
- `Update` button (below top rows)

#### Compare group fields
- `Compare on` toggle
- compare parameter dropdown
- compare values input (comma-separated)
- `Clear` button
- active list summary lines: `Parameter: [v1, v2, ...]`

Rules:
- when compare is turned on, set animation off automatically (user may re-enable manually)
- build Cartesian combinations across active compare parameter lists
- if `Policy` is in compare list, start each run from that policy defaults; only explicitly compared values override

#### General group fields
- `Max steps`
- `Episodes`
- `Epsilon max`
- `Epsilon decay`
- `Epsilon min`
- `Gamma`

#### Specific group fields (visible order)
- top row: `Policy` dropdown
- row order:
  1. `Hidden layer` | `Activation`
  2. `LR` | `LR strategy`
  3. `Min LR` | `LR decay`
  4. `Replay size` | `Batch size`
  5. `Learning start` | `Learning frequency`
  6. `Target update` (single field row)

Notes:
- learning-rate text inputs use scientific notation formatting
- policy change applies policy defaults to relevant fields

#### Live Plot group fields
- `Moving average values` (default: `20`)

### Controls row
Exactly 8 equal-width buttons (left to right):
1. `Run single episode`
2. `Train and Run`
3. `Pause` / `Run`
4. `Reset All`
5. `Clear Plot`
6. `Save samplings CSV`
7. `Save Plot PNG`
8. `Current device: CPU/GPU`

### Current Run panel
- Steps progress bar
- Episodes progress bar
- status line format:
  - `Epsilon: <...> | LR: <...> | Best reward: <...> | Render: <off|on|skipped|idle>`

### Live Plot panel
- no title
- plot reward and moving average with same color
- eval checkpoints as separate style (`eval`)
- legend outside right side
- interactive legend toggling (text and handle clickable)
- preserve previous runs until `Clear Plot`

---

## Runtime behavior and responsiveness
- training runs in background thread(s)
- Tk widgets updated only in main thread
- worker threads write pending state only (thread-safe)
- periodic UI pump consumes pending updates
- render tick in main thread at configured FPS
- no worker-side GUI drawing
- throttled plot updates
- compare mode supports multiple runs with bounded parallelism
- latest-frame-wins rendering: do not queue all frames; always display newest available frame
- optional fast non-render evaluation path for non-visual episodes/checkpoints
- debounce resize-heavy UI updates (~100 ms) and avoid full relayout per configure event
- avoid repeated expensive redraws when state/visibility did not change

### Update-rate behavior
Use `Update rate (episodes)` as gating interval:
- reward data stored every episode internally
- animation updates on every Nth episode (and final episode)
- live plot refresh on every Nth episode (and final episode)

---

## Compare mode behavior
- each combination is one run
- no duplicate finalize lines
- immutable per-run metadata for labels
- selected policy determines which compare run drives render/status/progress when available

---

## Styling baseline (dark mode)
- theme preference: `clam`, fallback `vista`
- palette:
  - main bg `#1e1e1e`
  - panel bg `#252526`
  - input bg `#2d2d30`
  - text `#e6e6e6`
  - muted `#d0d0d0`
  - accent `#0e639c`
- keep consistent spacing tokens
- preserve equal-width control buttons

---

## Exports
- CSV export to `results_csv/`
- PNG export to `plots/`
- PNG filename encodes run params + timestamp

---

## Tests
- logic tests: environment/trainer/run_episode/evaluate core behavior
- GUI smoke tests: startup, clear/reset safety, plotting/legend interactions, compare finalization consistency
- run with `pytest`

Testing robustness rules:
- if Box2D/LunarLander is unavailable on a machine, skip affected tests with explicit reason instead of hard-failing whole suite
- include at least one headless smoke path for training loop + event propagation
- verify pause/resume/cancel transitions and final status reporting for worker jobs

---

## Requirements file baseline
Include at least:
- `stable-baselines3`
- `sb3-contrib`
- `torch`
- `matplotlib`
- `pillow`
- `pytest`

---

## Reproducibility rule
If File B provides only:
- project name
- environment details
- policy list
- `use SB3`

then this file must still be sufficient to regenerate a functionally equivalent project structure, behavior, and GUI.
