Goal
----
Set up a general reinforcement learning project including the project structure, core modules, behavior, GUI, tests, dependencies, and run instructions. By the user, this initial file will be combined with project specific instructions in a second markdown file.

High-level summary
------------------
- Use `object oriented programming`.
- Use `Pytorch` with GPU optimization for any deep learning approaches
- Environment in which an agent is trained.
- One or several agents using different learning polycies.
- Trainer that can run single episodes, train for multiple episodes, optionally save sampled transitions to CSV, and support per-step progress callbacks.
- A Tkinter GUI that visualizes the environment and agent; provides controls to run single episodes, train, save CSV/PNG, clear plot and embed a matplotlib live plot of episode rewards.

Files & layout
--------------
- Use the project-name given by the user to create the following files within the current project directory.
- `project-name_app.py` - entry point, initializing all classes, runs GUI
- `project-name_logic.py` — core environment and agent implementations, plus trainer.
- `project-name_gui.py` — full GUI: environment, controls, `Current State` panel, `Training Parameters` panel, live matplotlib plot, image rendering using Pillow when available, CSV/PNG saving, and threading for training.
- `tests/test_project-name_logic.py` — unit tests for environment/agents/trainer.
- `requirements.txt` — list runtime dependencies (matplotlib, pillow, pytest optional).
- `results_csv/` & `plots/` — output folders for CSVs and PNGs.
- README markdown file describing implementation and how to run.

Functional requirements (detailed)
---------------------------------
Environment & logic:
- Environment, agent and policy depend on the project specifics defined in the second markdown file.

- `Trainer`:
  - `run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)`;
    calls `progress_callback(step)` each step if provided.
  - `train(policy, num_episodes, max_steps, epsilon, save_csv=None)` runs many episodes, optionally writing sampled transitions to a CSV in `results_csv/`.

GUI behavior and layout:
- Main window using a grid with two columns split into labeled panels (ttk.LabelFrame) as follows (top → bottom):
  - `Environment` (top row, full width): specifics will be defined in second markdown file.
  - `Controls` (left column): a 2×3 grid of buttons, arranged as:
    - Row0: `Reset All` | `Clear Plot`
    - Row1: `Run single episode` | `Save samplings CSV`
    - Row2: `Train and Run` | `Save Plot PNG`
    Buttons should expand horizontally in their cells (sticky='ew').
    Additional buttons may be defined in the second markdown file.
  - `Current State` (below Controls): single-line status label formatted as space-padded string (padded to the length of the string `Training`) showing `Training:`/`Idle:` then `step` and `episode` counters formatted as space-padded 4-digit numbers e.g. `Training: step:    3  episode:   12` or `    Idle: step:    0  episode:    0`.
  - `Training Parameters` (right column, tight width, spans Controls+Current State rows, do not expand horizontally): parameter inputs for max steps, episodes, policy dropdown, `Live plot` checkbox (default True), `reduced speed` checkbox (default True) next to live plot.  Additional parameters may be defined in the second markdown file.
  - `Live Plot` (bottom, full width): embedded matplotlib canvas that plots episode rewards for past runs and current run.
    - add legend outside right of the plot
    - entry name should state the policy
    - add toggle boxes for every legend entry; when an entry is untoggled the corresponding plot's visibility is set to false.

Performance and Responsiveness:
- Training must run on a background thread. GUI updates use `after(0, ...)` to schedule work on the main thread.
- Per-step UI updates must not trigger full canvas redraws; the `Current State` label should be updated directly.
- Throttle plot updates to ~150 ms using `self._last_plot_update` timestamp.
- Optionally, a `reduced speed` toggle causes a 33 ms sleep between episodes to make training progress observable.

User interactions
-----------------
- Controls:
  - `Run single episode` resets agent to configured start, runs `Trainer.run_episode(policy, ...)` once and animates the episode transitions sequentially; set `current_episode` to 1 and `current_step` to 0 before running.
  - `Train and Run` executes in a background thread; for each episode:
    - update `current_episode` via `after(0, ...)` helper so GUI label shows progress,
    - pass a `progress_callback` to `Trainer.run_episode` so GUI receives per-step callbacks; schedule per-step updates to the GUI thread with `after(0, ... self._set_current_counters(ep, step))` so `step` updates live,
    - append total reward to `rewards` list and (if `live_plot` active) schedule a plot update with `after(0, lambda: _update_plot(current_rewards=rewards.copy()))`.
    - if `reduced speed` is enabled, sleep 0.033 seconds between episodes to slow progress for observation.
  - `Reset All` requests stop (set `_stop_requested`), clears plot runs and map, and resets `agent_instance`.
  - `Save samplings CSV` calls `Trainer.train(..., save_csv=base)` or equivalent to produce CSV in `results_csv/`.
  - `Save Plot PNG` writes the embedded matplotlib figure to `plots/` with a filename encoding parameters: policy, eps min/max, alpha, gamma, episodes, max_steps, and a timestamp.

Testing
-------
- Provide unit tests in `tests/test_project-name_logic.py` that exercise `step`, `is_reachable`, `Trainer.run_episode` for different policies, and basic learning updates.
- Use `pytest` to run tests, ensure tests pass.

Dependencies & environment
--------------------------
- Python 3.8+ recommended.
- Runtime dependencies: `matplotlib`, `pillow` (optional but recommended), `pytest` (dev/test).
- Include a `requirements.txt` listing these packages.
- do not change the used Python environment