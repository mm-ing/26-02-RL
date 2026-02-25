# Goal
Set up a general reinforcement learning project including the project structure, core modules, behavior, GUI, tests, dependencies, and run instructions. By the user, this initial file will be combined with project specific instructions in a second markdown file.

# High-level summary
- Use `object oriented programming`.
- Use `Pytorch` with GPU optimization for any deep learning approaches
- Environment in which an agent is trained.
- One or several agents using different learning polycies.
- Trainer that can run single episodes, train for multiple episodes, optionally save sampled transitions to CSV, and support per-step progress callbacks.
- A Tkinter GUI that visualizes the environment and agent; provides controls to run single episodes, train, save CSV/PNG, clear plot and embed a matplotlib live plot of episode rewards.

# Files & layout
- Use the project-name given by the user to create the following files within the current project directory.
- `project-name_app.py` - entry point, initializing all classes, runs GUI
- `project-name_logic.py` — core environment and agent implementations, plus trainer.
- `project-name_gui.py` — full GUI: environment, controls, `Current State` panel, `Training Parameters` panel, live matplotlib plot, image rendering using Pillow when available, CSV/PNG saving, and threading for training.
- `tests/test_project-name_logic.py` — unit tests for environment/agents/trainer.
- `requirements.txt` — list runtime dependencies (matplotlib, pillow, pytest optional).
- `results_csv/` & `plots/` — output folders for CSVs and PNGs.
- README markdown file describing implementation and how to run.

# Functional requirements (detailed)
## Environment & logic:
- Environment, agent and policy depend on the project specifics defined in the second markdown file.

- `Trainer`:
  - `run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)`;
    calls `progress_callback(step)` each step if provided.
  - `train(policy, num_episodes, max_steps, epsilon, save_csv=None)` runs many episodes, optionally writing sampled transitions to a CSV in `results_csv/`.

## GUI behavior and layout:
- Main window using a grid with two columns split into labeled panels (ttk.LabelFrame) as follows (top → bottom):
  - `Environment` (top row, left column, height should be floatable): specifics will be defined in second markdown file.
  - `Parameters` (right column, tight width, spans `Environment` rows, do not expand horizontally, scrollable if needed): 
    - sort into following groups:
      - `Environment` (arrange in two columns, top to bottom)
        - `animation FPS` (refresh rate for envrionment animation in frames per second, default 10)
      - `General`
        - policiy dropdown menu 
        - parameter inputs (arrange in two columns, top to bottom):
          - `max steps`
          - `episodes`
          - `epsilon max` (when applicable)
          - `epsilon decay` (when applicable)
          - `epsilon min` (when applicable)
      - `Specific` (specific parameters for chosen policy)
        - parameter inputs (arrange in two columns, top to bottom):
          - `gamma` (when applicable, default 0.99)
          - `learning rate`(when applicable, default 0.001)
          - `replay size` (when applicable, default 50000)
          - `batch size`(when applicable, default 64)
          - `target update` (when applicable, default 200)
          - `hidden layer size` (when applicable, default 128)
      - `Live Plot`
        - parameter inputs (arrange in two columns, top to bottom):
          - `moving average values` (default 20)
    - Additional parameters may be defined in the second markdown file.
  - `Controls` (full width, below `Environment` and `Parameters`):
    - one row of buttons, all buttons with equal width, arrange left to right
      - `Run single episode`
      - `Train and Run`
      - `Pause`/`Run` (allows pausing the training, states `Pause` when running, states `Run` when paused)
      - `Reset All`
      - `Clear Plot`
      - `Save samplings CSV`
      - `Save Plot PNG`
    - Buttons should expand horizontally in their cells (sticky='ew').
    - Additional buttons may be defined in the second markdown file.
  - `Current Run` (full width, below Controls, arrange top to bottom): 
    - progress bar for steps (filling left to right), labeled `Steps`, label left
    - progress bar for episodes (filling left to right), labeled `Episodes`, label left
    - single-line status `Epsilon` counter formatted as 1.2-digit float, e.g. "Epsilon: 0.90"
  - `Live Plot` (bottom, full width, height should be floatable): 
    - embedded matplotlib canvas that plots episode rewards for past runs and current run
      - plot live reward
      - plot the moving average of the live reward (default: average over 20 values, use input from `Training Params`)
      - both plots use the same colour
      - line style: live reward thin line, slightly transparent; moving average bold line
    - add legend outside right of the plot
    - entry name should state the policy and relevant parameters
    - legend entries must be interactive toggles (click legend label/line to show or hide corresponding plot line)
    - do not add a separate toggle/checkbox panel for legend visibility
    - hidden legend entries should be visually de-emphasized (e.g. lower alpha)

## Performance and Responsiveness:
- Training must run on a background thread.
- Worker thread must not update Tk widgets directly.
- Use a periodic main-thread UI pump (~20–50 ms) that coalesces and applies latest pending updates.
- Worker thread writes only latest pending state (e.g. episode/step counters, frame-refresh request, rewards snapshot, finalize-run flag) into thread-safe shared state.
- Per-step UI updates must not trigger full canvas redraws; the `Current State` label should be updated directly.
- Throttle plot updates to ~150 ms using `self._last_plot_update` timestamp.
- Optionally, a `reduced speed` toggle causes a 33 ms sleep between episodes to make training progress observable.
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
  - `Pause`/`Run`
    - allows pausing the training
  - `Reset All` requests stop (set `_stop_requested`), clears plot runs and map, and resets `agent_instance`.
    - must not crash while background training is active.
  - `Clear Plots` clears all plots from the live plot
  - `Save samplings CSV` calls `Trainer.train(..., save_csv=base)` or equivalent to produce CSV in `results_csv/`.
  - `Save Plot PNG` writes the embedded matplotlib figure to `plots/` with a filename encoding parameters: policy, eps min/max, learning rate, gamma, episodes, max_steps, and a timestamp.

# Testing
- Provide unit tests in `tests/test_project-name_logic.py` that exercise `step`, `is_reachable`, `Trainer.run_episode` for different policies, and basic learning updates.
- Use `pytest` to run tests, ensure tests pass.
- Add GUI smoke checks for responsiveness and robustness:
  - long run with `reduced speed=False` and `live_plot=True` should remain responsive.
  - repeated `Clear Plot` and `Reset All` during training should not raise exceptions.
  - interactive legend toggling should still work after multiple runs and clear/reset cycles.

# Dependencies & environment
- Python 3.8+ recommended.
- Runtime dependencies: `matplotlib`, `pillow` (optional but recommended), `pytest` (dev/test).
- Include a `requirements.txt` listing these packages.
- do not change the used Python environment