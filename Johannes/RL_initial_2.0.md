# Goal
Set up a general reinforcement learning project including the project structure, core modules, behavior, GUI, tests, dependencies, and run instructions. By the user, this initial file will be combined with project specific instructions in a second markdown file.

# High-level summary
- Use `object oriented programming`.
- Use `PyTorch` for deep learning approaches with runtime-selectable device execution (CPU default, optional GPU when available).
- Environment in which an agent is trained.
- One or several agents using different learning policies.
- Trainer that can run single episodes, train for multiple episodes, optionally save sampled transitions to CSV, and support per-step progress callbacks.
- A Tkinter GUI that visualizes the environment and agent; provides controls to run single episodes, train, save CSV/PNG, clear plot and embed a matplotlib live plot of episode rewards.

# Files & layout
- Use the project-name given by the user to create the following files within the current project directory.
- `project-name_app.py` - entry point, initializing all classes, runs GUI
- `project-name_logic.py` — core environment and agent implementations, plus trainer.
- `project-name_gui.py` — full GUI: environment, controls, `Current Run` panel, `Parameters` panel, live matplotlib plot, image rendering using Pillow when available, CSV/PNG saving, and threading for training.
- `tests/test_project-name_logic.py` — unit tests for environment/agents/trainer.
- `requirements.txt` — list runtime dependencies (matplotlib, pillow, pytest optional).
- `results_csv/` & `plots/` — output folders for CSVs and PNGs.
- README markdown file describing implementation and how to run.

# Functional requirements (detailed)
## Environment & logic:
- Environment, agent and policy depend on the project specifics defined in the second markdown file.
- specific parameters for each policy have to be adjusted and stored (internally) separately
- if a policy display label is not a valid Python identifier (for example contains `+`), use a valid internal class name and map it to the requested display label in GUI/legend/export naming.
- if the project-specific policy list is explicitly constrained, expose exactly that list in GUI and training flows (no extra baseline policy).

- `Trainer`:
  - `run_episode(policy, epsilon=<project-specific default>, max_steps=<project-specific default>, progress_callback=None)`;
    calls `progress_callback(step)` each step if provided.
  - training uses the same raw environment reward signal as returned by the environment step function (no reward shaping).
  - learning updates are controlled by policy parameters `replay warmup` and `learning cadence`.
  - network activation is controlled by policy parameter `activation function`.
  - `train(policy, num_episodes, max_steps, epsilon, save_csv=None)` runs many episodes, optionally writing sampled transitions to a CSV in `results_csv/`.

## GUI behavior and layout:
- Main window using a grid with two columns split into labeled panels (ttk.LabelFrame) as follows (top → bottom):
  - `Environment` (top row, left column, height should be floatable): specifics will be defined in second markdown file.
    - rendered environment image/frame should be responsive to available panel width, preserve aspect ratio, and avoid left/right cutoffs.
    - use a dedicated render canvas; center the frame in the canvas.
    - scale each frame to the maximum available canvas area while preserving aspect ratio.
    - redraw on panel/canvas resize.
  - `Parameters` (right column, tight width, spans `Environment` rows, do not expand horizontally, scrollable if needed): 
    - within each parameter group, place inputs in 4 columns using the pattern `label, input, label, input` (two fields per row).
    - sort into following groups:
      - `Environment`
        - top row contains:
          - `animation on` toggle (default `on`) to activate/deactivate environment animation
          - `animation FPS` (refresh rate for environment animation in frames per second, default 10)
        - `update` button which resets the environment based on current settings and is placed below the top row
      - `Compare`
        - `compare on` toggle (default `off`)
        - add a compare-parameter dropdown (must include policy and other configurable run parameters)
        - add a compare-values input next to the dropdown; values are entered as comma-separated lists (e.g. `option1, option2`), spaces are ignored
        - for selectable/enum parameters (e.g. policy), each option may appear at most once in the list
        - below the controls, render active compare lists one parameter per line as: `Parameter name: [value1, value2]`
        - this group is placed between `Environment` and `General`
        - when `compare on` is activated, `animation on` is automatically set to `off`; user may manually set it to `on` afterwards
        - if `Policy` is part of compare parameters, each generated run must start from that policy's default hyperparameter set; only explicitly compared parameters may override those defaults.
      - `General`
        - parameter inputs:
          - `policy` (dropdown; changing policy applies policy-specific defaults)
          - `max steps` (default is project-specific)
          - `episodes` (default is project-specific)
          - `epsilon max` (when applicable)
          - `epsilon decay` (when applicable)
          - `epsilon min` (when applicable)
        - changing the selected policy applies policy-specific default values to relevant general/specific fields.
      - `Specific` (specific parameters for chosen policy)
        - parameter inputs:
          - `hidden layer size` (when applicable, default is policy/project-specific; accepts comma-separated lists such as `256,256`)
          - `activation function` (when applicable, dropdown, default is policy/project-specific; options: `ReLU`, `Tanh`, `LeakyReLU`, `ELU`)
          - `learning rate`(when applicable, default is policy/project-specific; input/output should use scientific notation)
          - `gamma` (when applicable, default is policy/project-specific)
          - `LR strategy` (when applicable, dropdown, default is policy/project-specific; options: `exponential`, `linear`, `cosine`, `loss-based`, `guarded natural gradient`)
          - `LR decay` (when applicable, default is policy/project-specific; interpreted by selected strategy)
          - `min learning rate` (when applicable, default is policy/project-specific; input/output should use scientific notation)
          - `learning cadence` (when applicable, default is policy/project-specific)
          - `replay size` (when applicable, default is policy/project-specific)
          - `batch size`(when applicable, default is policy/project-specific)
          - `target update` (when applicable, default is policy/project-specific)
          - `replay warmup` (when applicable, default is policy/project-specific)
          - `GAE λ` (when applicable, default is policy/project-specific)
          - `PPO clip` (when applicable, default is policy/project-specific)
      - `Live Plot`
        - parameter inputs:
          - `moving average values` (default 20)
    - Additional parameters may be defined in the second markdown file.
    - parameter content area must stretch vertically when the window height changes.
    - vertical scrollbar is shown only when content height exceeds visible panel height.
    - hide scrollbar when all controls fit in the visible area.
    - while the scrollbar is visible, controls in the right-most input column must not be clipped/cut off; reserve effective width for scrollbar and reduce input widget widths as needed.
    - mouse-wheel scrolling is active only when cursor is inside the parameters panel and scrollbar is visible.
    - in dark mode, toggle/checkbutton controls must not visually brighten their field background on hover.
  - `Controls` (full width, below `Environment` and `Parameters`):
    - one row of buttons in exactly 8 equal-width grid columns, arrange left to right
      - `Run single episode`
      - `Train and Run`
      - `Pause`/`Run` (allows pausing the training, states `Pause` when running, states `Run` when paused)
      - `Reset All`
      - `Clear Plot`
      - `Save samplings CSV`
      - `Save Plot PNG`
      - `Current device: CPU/GPU` (button for runtime device selection)
    - Buttons should expand horizontally in their cells (sticky='ew').
    - Additional buttons may be defined in the second markdown file.
    - each `Train and Run` starts from a freshly initialized network for each trained policy (new weights; no carry-over from prior runs).
    - `Reset All` must not trigger matplotlib `UserWarning` messages when no labeled plot lines exist (e.g., empty legend state).
  - `Current Run` (full width, below Controls, arrange top to bottom): 
    - progress bar for steps (filling left to right), labeled `Steps`, label left
    - progress bar for episodes (filling left to right), labeled `Episodes`, label left
    - single-line status text in format `Epsilon: <value> | LR: <value> | Current x: <value or n/a> | Best x: <value or n/a>`
    - `LR` in this status line should display the live optimizer learning rate of the active run/policy (not only the static UI input value) and should be formatted in scientific notation.
    - `Best x` tracks best position within the current run and resets on `Run single episode`, `Train and Run`, and `Reset All`.
    - keep label column fixed width and progress-bar column stretchable so bars use the full available width.
    - steps progress uses the current episode effective max steps so the bar reaches full width on episode completion.
  - `Live Plot` (bottom, full width, height should be floatable): 
    - embedded matplotlib canvas that plots episode rewards for past runs and current run
      - no plot title
      - plot live reward
      - plot the moving average of the live reward (default: average over 20 values, use input from `Training Params`)
      - track and plot deterministic evaluation reward checkpoints (periodic eval episodes without exploration and without learning updates), shown as a separate line style.
      - deterministic eval tracking is active by default in training flows (including compare mode); use periodic checkpoints (default every 10 episodes, averaged over 3 deterministic eval episodes).
      - both plots use the same colour
      - line style: live reward thin line, slightly transparent; moving average bold line
      - each `Train and Run` appends a new run (reward + moving average); previous runs remain visible until `Clear Plot`
    - add legend outside right of the plot
    - legend base label format must be: `<policy> | eps(<epsilon max>/<epsilon min>) | lr=<learning rate> | lr-strategy=<LR strategy> | lr-decay=<LR decay> | min-lr=<min learning rate>`
    - LR-related legend fields (`lr`, `lr-decay`, `min-lr`) should be rendered in scientific notation.
    - each run contributes exactly two legend entries: `<base label> | reward` and `<base label> | MA`
    - each run contributes one additional eval legend entry when eval points exist: `<base label> | eval`.
    - legend labels must be built from immutable per-run metadata captured at run start (policy, epsilon settings, LR settings, moving-average window), not from mutable current UI values.
    - if a run is paused/stopped and a new `Train and Run` starts, previously running/finalized legend entries must keep their original metadata and must not be overwritten by the new run parameters.
    - in compare mode, each run finalization must use its own captured per-run metadata (no fallback to current UI field values).
    - legend entries must be interactive toggles (click legend label text or legend line/marker to show or hide corresponding plot line)
    - in compare mode, legend entries should be visible during active training (not only after run finalization)
    - do not add a separate toggle/checkbox panel for legend visibility
    - hidden legend entries should be visually de-emphasized (e.g. lower alpha)

  ## Visual polish (minimal):
  - keep the same functional layout and behavior, but apply a minimal modern ttk styling pass.
  - use ttk theme selection in this order to match current implementation: prefer `clam`, fallback to `vista`.
  - use consistent spacing tokens across frames/controls (uniform outer and inner paddings).
  - keep one-row control buttons equal in size and visually consistent.
  - use dark mode as default.
  - use the following dark palette tokens for reproducibility:
    - main background: `#1e1e1e`
    - panel background: `#252526`
    - input background: `#2d2d30`
    - foreground text: `#e6e6e6`
    - muted text: `#d0d0d0`
    - accent: `#0e639c`
  - button style colors:
    - default `TButton`: background `#3a3d41`, active `#4a4f55`, pressed `#2f3338`
    - highlighted training button `Primary.TButton`: background `#0e639c`, active `#1177bb`, pressed `#0b4f7a`
    - highlighted pause button `Pause.TButton`: background `#a66a00`, active `#bf7a00`, pressed `#8c5900`
  - progressbar style: trough `#343434`, fill/accent `#0e639c`.
  - environment render canvas background: `#111111`.
  - live plot styling:
    - figure background `#1e1e1e`, axes background `#252526`
    - x/y labels color `#e2e2e2`, ticks `#dddddd`, spines `#9a9a9a`
    - legend space reserved on the right (`subplots_adjust(right=0.75)`).
  - control-button highlight state must be:
    - idle/start-up: no highlighted button
    - while training is running: highlight `Train and Run`
    - while paused: highlight `Pause` and remove highlight from `Train and Run`

## Performance and Responsiveness:
- Training must run on a background thread.
- Worker thread must not update Tk widgets directly.
- Worker thread must not read Tk variables directly; snapshot required UI values on the main thread before the worker starts.
- Use a periodic main-thread UI pump (~20–50 ms) that coalesces and applies latest pending updates.
- Worker thread writes only latest pending state (e.g. episode/step counters, rewards snapshot, finalize-run flag, immutable run metadata snapshot) into thread-safe shared state.
- worker thread must not render/draw GUI images.
- environment access (`reset`, `step`, `render`) must be serialized/thread-safe.
- main thread performs environment frame rendering and drawing at the configured animation FPS.
- if animation is deactivated (`animation on` off), skip environment animation drawing work.
- Per-step UI updates must not trigger full canvas redraws; the `Current Run` status label should be updated directly.
- Throttle plot updates to ~150 ms using `self._last_plot_update` timestamp.
- live plot redraw should run only when new reward data points are available.
- in compare mode, allow a less frequent plot throttle (e.g. ~300 ms) for responsiveness.
- avoid deleting/recreating the render-canvas image item every frame; update an existing image item when possible.
- for live plotting, prefer incremental line data updates over full axis clear/redraw on every tick.
- environment `render` should avoid blocking UI (e.g. non-blocking lock attempt + cached last frame fallback).
- learning should start only after replay buffer warmup threshold is reached, then run on configured learning cadence.
- deterministic evaluation episodes must not mutate training state (no replay insertions, no optimizer updates, no end-of-episode learning flush).
- `Reset All` and `Clear Plot` must be safe during and after training, without crashes.

# User interactions
- Controls:
  - `Run single episode` 
    - resets agent to configured start position yet keeps last trained policy if available
    - runs `Trainer.run_episode(policy, ...)` once and animates the episode transitions sequentially
  - `Train and Run` executes in a background thread; for each episode:
    - update `current_episode` and `current_step` through coalesced pending state consumed by the UI pump,
    - pass a `progress_callback` to `Trainer.run_episode` so per-step progress updates pending state,
    - append total reward to `rewards` list and (if `live_plot` active) update latest rewards snapshot for throttled UI-pump plotting
    - environment animation must be refreshed by the main thread on FPS cadence using the latest environment state
  - compare mode behavior (`compare on` = true):
    - `Train and Run` starts parallel training runs for every Cartesian combination of the active compare-parameter lists.
    - each parameter combination contributes its own run to live plot.
    - exactly one finalized run per parameter combination per compare launch (no duplicate extra run).
    - for each combination, any parameter not listed in compare lists uses the current value from the main input fields.
    - environment animation, status line, and progress bars reflect the currently selected policy in the dropdown (matching run when available).
  - `Pause`/`Run`
    - allows pausing the training
  - `Reset All` requests stop (set `_stop_requested`), clears plot runs and map, and resets `agent_instance`.
    - must not crash while background training is active.
  - `Clear Plot` clears all plots from the live plot
  - `Save samplings CSV` calls `Trainer.train(..., save_csv=base)` or equivalent to produce CSV in `results_csv/`.
  - `Save Plot PNG` writes the embedded matplotlib figure to `plots/` with a filename encoding parameters: policy, eps min/max, learning rate, gamma, episodes, max_steps, and a timestamp.
  - learning-rate tokens encoded in PNG filenames should use scientific notation.

# Testing
- Provide unit tests in `tests/test_project-name_logic.py` that exercise `step`, `is_reachable`, `Trainer.run_episode` for different policies, and basic learning updates.
- Use `pytest` to run tests, ensure tests pass.
- Add GUI smoke checks for responsiveness and robustness:
  - long run with `live_plot=True` should remain responsive.
  - repeated `Clear Plot` and `Reset All` during training should not raise exceptions.
  - interactive legend toggling should still work after multiple runs and clear/reset cycles.
  - pausing a run and starting a new `Train and Run` must preserve old legend metadata and append a distinct new run label.
  - in compare mode, finalized legend labels must match per-policy captured metadata even if current UI values were changed during/after training.
  - in compare mode, finalized legend labels must match per-run captured metadata even if current UI values were changed during/after training.

# Dependencies & environment
- Python 3.8+ recommended.
- Runtime dependencies: `matplotlib`, `pillow` (optional but recommended), `pytest` (dev/test).
- Include a `requirements.txt` listing these packages.
- do not change the used Python environment
