Project reconstruction prompt — GridWorld RL

Goal
----
Provide a complete prompt that enables recreating the GridWorld Reinforcement Learning project from scratch. The prompt should describe the project structure, core modules, behavior, GUI, tests, dependencies, and run instructions so an engineer or an LLM can reimplement the project to match the existing repository.

High-level summary
------------------
- A small tabular GridWorld environment (grid with blocked cells, start and target) and several tabular learning agents:
  - `QLearningAgent` (Q(s,a) updates)
  - `MonteCarloAgent` (first-visit Monte Carlo state-value V(s))
  - `SARSAAgent` and `ExpectedSarsaAgent` (on-policy and expected variants)
- `Trainer` that can run single episodes, train for multiple episodes, optionally save sampled transitions to CSV, and support per-step progress callbacks.
- A Tkinter GUI (`gridworld_gui.py`) that visualizes the grid, agent, target, blocked cells and agent path; provides controls to run single steps/episodes, train, view tables, save CSV/PNG, and embed a matplotlib live plot of episode rewards.

Files & layout
--------------
- `gridworld_logic.py` — core environment and agent implementations, plus `Trainer`.
- `gridworld_gui.py` — full GUI: scrollable Canvas map, controls, `Current State` panel, `Training Parameters` panel, live matplotlib plot, image rendering using Pillow when available, CSV/PNG saving, and threading for training.
- `tests/test_gridworld_logic.py` — unit tests for environment/agents/trainer.
- `requirements.txt` — list runtime dependencies (matplotlib, pillow, pytest optional).
- `results_csv/` & `plots/` — output folders for CSVs and PNGs.
- README and helper markdown files (`GUI_initial.md`, `RL_initial.md`) describing implementation and how to reconstruct.

Functional requirements (detailed)
---------------------------------
Environment & logic:
- `Grid` class:
  - Attributes: `M` cols, `N` rows, `blocked` set, `start`, `target`.
  - Methods: `in_bounds`, `is_blocked`, `valid`, `neighbors`, `is_reachable(start=None, target=None)`, `step(s, action, noise=0.0)`.
  - `ACTIONS` mapping 0: up, 1: down, 2: left, 3: right.

- Agents:
  - `QLearningAgent(grid, alpha, gamma)` with `Q` table, `get_q`, `best_action`, `update`.
  - `MonteCarloAgent(grid, gamma)` estimating `V` via first-visit MC (`process_episode`) and `best_action` derived from neighbor V(s).
  - `SARSAAgent` and `ExpectedSarsaAgent` implement state-action methods and updates.

- `Trainer(grid, noise=0.0)`:
  - `run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)` supports SARSA/Expected SARSA and Q/MC loops; returns transitions and total reward; calls `progress_callback(step)` each step if provided.
  - `train(policy, num_episodes, max_steps, epsilon, save_csv=None)` runs many episodes, optionally writing sampled transitions to a CSV in `results_csv/`.

GUI behavior and layout:
- Main window split into labeled panels (ttk.LabelFrame):
  - `Environment` (full width): left = scrollable Canvas drawing grid; right = environment inputs (grid M/N, start, target, blocked inputs, Apply/Reset).
  - `Controls` (left column): 2×4 button grid for step/episode/train/show/save/reset.
  - `Current State` (below Controls): single-line status label showing `Training:`/`Idle:` then `step` and `episode` counters formatted as space-padded 4-digit numbers e.g. `Training: step:    3  episode:   12`.
  - `Training Parameters` (right column, tight width, spans Controls+Current State rows): parameters for alpha, gamma, max steps, episodes, epsilon min/max, policy dropdown, `Live plot` checkbox (default True), `reduced speed` checkbox (default True) next to live plot.
  - `Live Plot` (bottom): embedded matplotlib canvas that plots episode rewards for past runs and current run.

Performance and Responsiveness:
- Training must run on a background thread. GUI updates use `after(0, ...)` to schedule work on the main thread.
- Per-step UI updates must not trigger full canvas redraws; the `Current State` label should be updated directly.
- Throttle plot updates to ~150 ms using `self._last_plot_update` timestamp.
- Optionally, a `reduced speed` toggle causes a 33 ms sleep between episodes to make training progress observable.

User interactions
-----------------
- Clicking on canvas maps to grid cells (taking scroll offset into account using `canvasx`/`canvasy`).
- Clicking/dragging agent/target shows preview images and updates positions on release (avoid placing on blocked cells).
- Toggling blocked cells must check `Grid.is_reachable()` and reject changes that block the target.
- Controls call appropriate methods: single-step accumulates transitions, single-episode runs `Trainer.run_episode` once and animates transitions, Train runs many episodes and updates plot live.
- Save CSV writes sampled transitions; Save Plot PNG writes plot to `plots/` with parameters in the filename.

Testing
-------
- Provide unit tests in `tests/test_gridworld_logic.py` that exercise `Grid.step`, `is_reachable`, `Trainer.run_episode` for different policies, and basic learning updates.
- Use `pytest` to run tests, ensure tests pass.

Dependencies & environment
--------------------------
- Python 3.8+ recommended.
- Runtime dependencies: `matplotlib`, `pillow` (optional but recommended), `pytest` (dev/test).
- Include a `requirements.txt` listing these packages.

Developer guidance for recreation
---------------------------------
- Write clean, modular code with helper methods for drawing and event handling.
- Use `Image`/`ImageDraw` + downsampling (LANCZOS) to create anti-aliased agent/target icons when Pillow available.
- Keep PhotoImage objects as attributes on the Tk root to avoid Tcl garbage-collection.
- Ensure `Trainer.run_episode` can accept an optional `progress_callback` for UI step updates.
- Use `threading.Thread(..., daemon=True)` for the training worker.
- Use `self.after(0, ...)` for all GUI updates coming from background threads.

Run instructions
----------------
From project root (package `Johannes.GridWorld`):

Run GUI (example):
```powershell
cd <project-root>/26-02-RL/Johannes/GridWorld
python gridworld_gui.py
```

Run tests:
```powershell
cd <project-root>/26-02-RL/Johannes/GridWorld
pytest -q
```

Acceptance criteria
-------------------
- The recreated project should provide the same user experience: identical panels, widgets, behaviors, and file outputs (`results_csv/*.csv`, `plots/*.png`).
- Training progress must be visible in the `Current State` label and the live plot, with performance safeguards (throttled plotting, minimal redraws).

End of reconstruction prompt.
