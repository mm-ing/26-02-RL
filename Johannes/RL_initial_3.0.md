# RL Master Blueprint (v2.1)

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
- publish structured events (step/episode/training_done/error) rather than direct widget calls

### App module
- simple entrypoint that creates `Tk()`, instantiates GUI, starts `mainloop()`.
- set startup environment guards **before importing GUI/ML modules**:
  - `TF_ENABLE_ONEDNN_OPTS=0`
  - `TF_CPP_MIN_LOG_LEVEL=3`
  to suppress noisy TensorFlow/oneDNN info logs in console launches.

---

## SB3 policy behavior
- Expose exactly the policy list from File B.
- Keep GUI display names even if internal mapping requires valid Python identifiers.
- Maintain per-policy default config snapshots.
- Reinitialize policy weights for each new `Train and Run` launch.
- Keep deterministic evaluation separate from exploration training episodes.
- Tune per-policy defaults for the target environment (do not reuse one-size-fits-all defaults).
- Keep on-policy settings internally consistent (for example `PPO`: use `n_steps >= batch_size` and prefer `batch_size` divisibility to avoid truncated mini-batch warnings).
- For off-policy defaults (`SAC`/`TD3`), use realistic replay warmup and buffer sizes (larger buffer + sufficient `learning_starts`) to improve early training stability.
- Runtime device is selectable as `CPU` or `GPU` for all exposed policies, with default `CPU`.

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
- canvas background baseline: `#111111`
- keep aspect ratio, center image
- redraw on resize
- render in main thread only

### Parameters panel
- scrollable content region
- parameter groups must expand to full available parameters-panel width (no fixed narrow content width)
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
- `Update` button
- environment-specific parameters below `Update`:
  - `healthy_reward`
  - `reset_noise_scale`

#### Compare group fields
- `Compare on` toggle
- top control row: `Compare on` at left, `Clear` and `Add` buttons at right (side-by-side)
- compare parameter dropdown (all `General` + `Specific` parameters; no `Environment` parameters)
- compare values text input (comma-separated)
- second row layout: `Parameter` dropdown (left) and `Values` input (right), without additional field labels
- compare values typed suggestions + `Tab` completion for categorical params (`Policy`, `Activation`, `LR strategy`)
- compare values completion preview hint appears below the `Values` input (for example: `Tab -> Tanh`)
- active list summary lines: `Parameter: [v1, v2, ...]`

Rules:
- when compare is turned on, set animation off automatically (user may re-enable manually)
- `Add` commits the selected parameter + current values text into active compare lists
- pressing `Enter` in compare values input triggers `Add`
- `Clear` removes all active compare parameter lists
- compare dropdown includes all parameters from `General` and `Specific`, and excludes `Environment`
- tab completion suggestions apply to categorical compare values and can be accepted with `Tab`
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
- within a group, input field widths are consistent (same width token)

#### Live Plot group fields
- `Moving average values` (default: `20`)
- `Show Advanced` toggle (default: `False`)
- when advanced section is shown: `Rollout full-capture steps` (default: `120`)
- when advanced section is shown: `Low-overhead animation` (default: `False`)

### Controls row
Exactly 8 equal-width controls (left to right):
1. `Run single episode`
2. `Train and Run`
3. `Pause` / `Run`
4. `Reset All`
5. `Clear Plot`
6. `Save samplings CSV`
7. `Save Plot PNG`
8. `Device` selector (`CPU` / `GPU`, default `CPU`)

Control highlight behavior:
- `Train and Run` highlight style is shown only while a training run is active
- `Pause/Run` highlight style is shown only while paused/activated
- while paused, `Train and Run` must revert to neutral/default style
- pressing `Train and Run` while paused must cancel the paused run and start a fresh run with current parameter-section values
- inactive state uses neutral/default button styling

### Current Run panel
- `Steps` label + progress bar
- `Episodes` label + progress bar
- steps progress semantics: advances only during replay animation playback (0→100 across playback frames), not from raw episode environment step count
- status line format:
  - `Epsilon: <...> | LR: <...> | Best reward: <...> | Render: <off|on|skipped|idle>`

### Live Plot panel
- no title
- axis labels: `x = Episodes`, `y = Reward`
- per-episode reward line is slightly transparent (recommended `alpha ~ 0.60`)
- `MA` and `eval` use the same color as per-episode reward
- `MA` and `eval` use `2x` line width of per-episode reward and different styles
- use `alpha = 1.0` for both `MA` and `eval` lines
  - recommended: `MA` dashed (`--`), `eval` dotted (`:`) with marker
- grid/spine/tick/label/legend text colors follow GUI text color baseline
- legend outside right side
- legend labels grouped per run: full parameter info shown once (reward line), with compact `moving average` / `evaluation rollout` entries
- use plain legend labels `moving average` and `evaluation rollout` (no run-id suffix)
- legend supports explicit line breaks in reward entry to prevent cutoff
- reward legend entry format:
  - line 1: `policy | steps=<max_steps> | gamma=<gamma>`
  - line 2: `epsilon=<...> | epsilon_decay=<...> | epsilon_min=<...>`
  - line 3: `LR=<...> | LR strategy=<...> | LR decay=<...>`
  - line 4: environment-specific params (`key=value | ...`)
- interactive legend toggling (text and handle clickable)
- legend hover affordance: on hover, legend entry appearance changes and cursor switches to hand to indicate clickability
- keep plot area left-aligned and reserve a fixed right gutter for legend
- preserve previous runs until `Clear Plot`

---

## Runtime behavior and responsiveness
- training runs in background thread(s)
- Tk widgets updated only in main thread
- worker threads write pending state only (thread-safe)
- periodic UI pump consumes pending updates
- before starting a new run, flush/consume queued worker events so pending finalize data is not lost
- starting a new run after `Pause` must preserve already finished run curves in live plot history
- tag worker events with run/session identifiers and ignore stale events from previous/canceled runs
- on reset/close, resume paused workers before stopping them to avoid shutdown hangs
- render tick in main thread at configured FPS
- no worker-side GUI drawing
- throttled plot updates
- compare mode supports multiple runs with bounded parallelism (max concurrent compare workers: `4`)
- latest-frame-wins rendering: do not queue all frames; always display newest available frame
- for visual update episodes, capture bounded per-step rollout frames and play them in GUI while training continues in background
- when `Low-overhead animation` is enabled, use reduced rollout frame budget, capped short full-capture threshold, and frame downsampling for lower rendering overhead
- short episodes should capture every step frame up to `Rollout full-capture steps`
- longer episodes should use adaptive stride sampling after that threshold (bounded frame budget)
- redraw render canvas only when a newer frame exists or when resize requires re-fit (avoid unconditional redraw every tick)
- reuse/update existing canvas image item for frames instead of deleting/recreating canvas content each tick
- optional fast non-render evaluation path for non-visual episodes/checkpoints
- debounce resize-heavy UI updates (~100 ms) and avoid full relayout per configure event
- avoid repeated expensive redraws when state/visibility did not change
- default/runtime device should be `CPU`, with selectable `CPU`/`GPU` for all policies
- see `Recent updates` (`2026-03-02`) for the runtime hotfix where `Animation on = False` must stop active replay immediately

### Update-rate behavior
Use `Update rate (episodes)` as gating interval:
- reward data stored every episode internally
- animation updates on every Nth episode (and final episode)
- live plot refresh on every episode (decoupled from update-rate gating)
- when animation updates are due, prefer bounded per-step rollout playback (sampled if needed for performance) instead of a single terminal frame
- for short episodes, avoid frame skipping before the full-capture threshold

---

## Compare mode behavior
- each combination is one run
- execute compare combinations with bounded parallel training (max `4` concurrent runs)
- for CPU compare runs with multiple workers, cap per-worker torch CPU thread count to reduce oversubscription and preserve effective parallel progress
- assign unique internal run IDs per compare combination so each run keeps an independent plot/history slot
- no duplicate finalize lines
- immutable per-run metadata for labels
- selected policy determines which compare run drives render/status/progress when available

---

## Styling baseline (dark mode)
- theme preference: `clam`, fallback `vista`
- visual/layout parity target: match the LunarLander GUI style profile unless File B overrides it
- palette:
  - main bg `#1e1e1e`
  - panel bg `#252526`
  - input bg `#2d2d30`
  - text `#e6e6e6`
  - muted `#d0d0d0`
  - accent `#0e639c`
- fonts:
  - default: `Segoe UI`, size `10`
  - group headings: `Segoe UI`, size `10`, bold
  - control buttons: `Segoe UI`, size `10`, bold
- keep consistent spacing tokens (recommended defaults):
  - outer pad `10`
  - inner pad `6`
  - tight pad `4`
  - label column min width `~92`
  - parameter input width `~9`
- preserve equal-width control buttons
- button styling baseline:
  - default dark button style for neutral actions (`bg #3a3d41`, active `#4a4f55`, pressed `#2f3338`)
  - accent style for active `Train and Run` (`#0e639c`, active `#1177bb`, pressed `#0b4f7a`)
  - amber style for active `Pause/Run` (`#a66a00`, active `#bf7a00`, pressed `#8c5900`)
- parameter panel baseline:
  - content fills full available panel width
  - avoid fixed/narrow canvas width constraints
- combobox listbox baseline:
  - list bg `#2d2d30`, list fg `#e6e6e6`, selected bg `#0e639c`, selected fg `white`
- progressbar baseline:
  - trough `#343434`, fill `#0e639c`
- plot styling baseline:
  - subtle grid enabled
  - spine tint/alpha aligned with dark theme
- subplot margins reserve right legend gutter (`left~0.04`, `right~0.78`)

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

Test execution isolation (required in multi-project workspaces):
- add local `pytest.ini` per project with `testpaths = tests`
- prefer isolated invocation: `python -m pytest -q --rootdir . --confcutdir . tests/...`
- avoid relying on workspace-root auto-discovery when sibling projects contain tests
- optional: include a local `run_tests.py` helper that runs the isolated command

Testing robustness rules:
- if Box2D/LunarLander is unavailable on a machine, skip affected tests with explicit reason instead of hard-failing whole suite
- if Tk/Tcl runtime assets are unavailable (for example missing `init.tcl`), GUI tests must `skip` with explicit reason instead of failing
- for GUI tests, guard both import-time and root-creation-time failures (`tk.Tk()`) before running assertions
- include at least one headless smoke path for training loop + event propagation
- verify pause/resume/cancel transitions and final status reporting for worker jobs
- include regression coverage for `Pause -> Train and Run` ensuring old finalized runs remain in plot history
- include regression coverage that stale worker events (for old session IDs) do not modify current progress/status/plot state
- ensure `run_episode` reports actual executed environment steps even when transition collection is disabled

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

---

## Recent updates

### 2026-03-03
- update-id: `COMPARE-UNIQUE-RUNID-PLOT-SYNC`
- synchronized compare runtime behavior with implementation/tests: each compare combination now gets a unique internal run ID, preventing live-plot/history overwrite when runs are created in rapid succession.

### 2026-03-03
- update-id: `COMPARE-PARALLEL-CPU-THREAD-TUNING-SYNC`
- synchronized compare runtime behavior with implementation: for multi-worker CPU compare runs, per-worker torch thread count is capped to reduce oversubscription and preserve effective parallel progress.

### 2026-03-03
- update-id: `GUI-PAUSE-RESTART-HIGHLIGHT-SYNC`
- synchronized control behavior with implementation: while paused, only `Pause/Run` is highlighted; pressing `Train and Run` during pause cancels the paused run and starts a fresh run from current parameters.

### 2026-03-03
- update-id: `RUNTIME-SESSION-GUARD-SYNC`
- synchronized runtime robustness spec with implementation/tests: worker events carry run/session tags and stale events from prior runs are ignored so old runs cannot overwrite active UI state.

### 2026-03-03
- update-id: `DEFAULTS-AND-DEVICE-README-SYNC`
- synchronized project docs to current baseline: runtime device selector supports `CPU`/`GPU` with default `CPU` and safe CPU fallback, and GUI default parameter profile reflects current PPO/SAC/TD3 tuned defaults.

### 2026-03-03
- update-id: `GUI-COMPARE-ROW-AND-HINT-ALIGN-SYNC`
- synchronized compare spec with current GUI implementation: top compare row now places `Compare on` left with `Clear`/`Add` controls right, input row has unlabeled side-by-side `Parameter` and `Values` fields, and completion hint is positioned directly below the `Values` field.

### 2026-03-03
- update-id: `GUI-COMPARE-LAYOUT-SIDE-BY-SIDE-SYNC`
- synchronized compare layout spec with current GUI: `Parameter` and `Values` fields are side-by-side, with `Add`/`Clear` controls side-by-side in the compare header row.

### 2026-03-03
- update-id: `GUI-COMPARE-DROPDOWN-TAB-PREVIEW-SYNC`
- synchronized compare spec with current GUI: compare parameter dropdown now includes all `General` + `Specific` parameters and excludes `Environment`; categorical compare values support typed suggestions, `Tab` completion, and live completion preview hint.

### 2026-03-03
- update-id: `GUI-PLOT-LINESTYLE-LEGEND-FORMAT-SYNC`
- synchronized live-plot spec with implementation: per-episode reward uses slight transparency (`alpha ~ 0.60`), `MA`/`eval` use same run color with `2x` width and distinct styles, and reward legend entry uses fixed multi-line parameter blocks to avoid legend cutoff.

### 2026-03-03
- update-id: `RUNTIME-SHUTDOWN-PAUSE-UNBLOCK-SYNC`
- synchronized runtime shutdown behavior with implementation/tests: on reset/close the GUI resumes paused workers before stop so blocked pause waits cannot keep the process alive after window close.

### 2026-03-03
- update-id: `GUI-ENV-FIELD-ORDER-WIDTH-SYNC`
- synchronized spec with current GUI: in the `Environment` group, `healthy_reward` and `reset_noise_scale` are positioned below the `Update` button; input/combobox field widths are standardized within each group via a shared width token.

### 2026-03-03
- update-id: `GUI-CURRENTRUN-PLOT-LABEL-SYNC`
- synchronized spec with current GUI: current-run progress bars are explicitly labeled `Steps` and `Episodes`; live plot uses fixed axis labels (`Episodes`, `Reward`) and plot chrome/text colors aligned with GUI text tone.

### 2026-03-03
- update-id: `GUI-COMPARE-AND-STYLE-SYNC`
- synchronized GUI spec with current implementation: compare group explicitly includes `Add` + `Clear` workflow for active parameter lists, environment canvas baseline `#111111`, and LunarLander-parity button/combobox/progressbar style values.

### 2026-03-03
- update-id: `APP-TF-ONEDNN-LOG-SUPPRESS`
- added app-entry startup rule: set `TF_ENABLE_ONEDNN_OPTS=0` and `TF_CPP_MIN_LOG_LEVEL=3` before GUI/runtime imports to avoid TensorFlow oneDNN informational startup warnings in terminal output.

### 2026-03-02
- update-id: `BW-ANIM-TOGGLE-RUNTIME-HOTFIX`
- fixed runtime behavior: setting `Animation on = False` must stop replay animation immediately for active runs (clear queued playback frames, reset replay progress, and propagate runtime animation settings to active trainer)
- added GUI regression coverage for this hotfix: toggle animation off clears replay queue/state and updates status render state to `off`
