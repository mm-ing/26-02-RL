Project: GridWorld RL (full reproduction prompt)

Goal
- Create a self-contained, runnable Python project implementing a small tabular GridWorld environment with two learning agents (Q-learning and Monte Carlo state-value), a simple Tkinter GUI for interactive control/visualization, plotting of training reward history (matplotlib embedded), CSV export of run data, and unit tests.

High-level requirements
- Environment: grid with dimensions M (columns) and N (rows). Coordinates are (x,y) with x in [0..M-1] left→right and y in [0..N-1] top→bottom. The agent starts at `start` and must reach `target`. Some cells can be `blocked` and are impassable. Grid must provide deterministic step() with optional slip/noise parameter.
- Rewards: reward = 0 on reaching `target`, otherwise reward = -1 per step.
- Agents:
  - `QLearningAgent` (tabular Q(s,a)) with learning rate `alpha` and discount `gamma` using TD update: Q(s,a) ← Q + α * (r + γ * max_a' Q(s',a') - Q).
  - `MonteCarloAgent` implementing first-visit, state-value Monte Carlo prediction/control (V(s)). It should accumulate episode states and rewards and, on episode completion, compute returns G and update V(s) by averaging first-visit returns for each state. Provide `get_v(s)` and `best_action(s)` which chooses the action that leads to the neighbor state with highest estimated V (ties broken randomly).
- Trainer: `Trainer` runs episodes using epsilon-greedy selection when `policy.best_action` exists. `Trainer.run_episode` should optionally update Q (if policy has `update`) during the episode and, if policy is MonteCarloAgent (state-value), after an episode call `policy.process_episode(episode_states, rewards)` where `episode_states` is [S0..ST] and `rewards` is [r0..r_{T-1}]. Provide `train()` which can write transitions to CSV into a local `results_csv` folder inside the package.
- GUI (Tkinter): a main window with a Canvas showing the grid (scales on resize), input controls for grid size, start/target, blocked cells (add/remove), policy selection (Q or MC), hyperparameters (alpha, gamma, epsilon default handling, max_steps, episodes), buttons: `Apply grid`, `Reset map`, `Run single step`, `Run single episode`, `Train and Run`, `Reset All`, `Show Value Table`, `Show Q-table`, `Save samplings CSV`. Canvas supports dragging agent/target, hover preview of cell under mouse, and clickable to toggle blocked cells with reachability check (prevent blocking start→target). Drawing: agent and target icons (Pillow recommended for anti-aliased images), path trace for agent (polyline with arrowheads), blocked cells colored.
- Single-step behavior: repeated presses of `Run single step` should accumulate into an ongoing episode — do not reset the accumulated episode between single steps; only when `Run single episode` or `Reset map` is pressed should accumulation clear. While accumulating allow Q updates immediately for Q-learning; when an episode finishes, call MonteCarloAgent.process_episode with full episode states+rewards.
- Value/Q display: `Show Q-table` displays state × actions table (treeview). `Show Value Table` displays V(s) for MonteCarloAgent or derived V(s)=max_a Q(s,a) for Q-learning.
- Plotting (matplotlib embedded): live plotting option during training; keep history of runs with distinct colors, clickable legend entries to toggle visibility, horizontal zero line shown and included in ticks.
- Threading: training must run in a background thread; GUI updates via `after()` to avoid concurrency issues. Provide a `_stop_requested` flag and a `Reset All` button that sets `_stop_requested` so training stops between episodes, clears `plot_runs`, resets map, and clears agent instance.
- Tests: unit tests for `Grid.step` behavior, blocked enforcement, and `QLearningAgent.update`. Ensure tests are runnable headlessly (avoid requiring a display); GUI tests may be minimal smoke tests only.

File layout (suggested)
- gridworld_logic.py
  - classes: Grid, QLearningAgent, MonteCarloAgent, Trainer.
- gridworld_gui.py
  - class GridWorldGUI(tk.Tk) with methods to build UI, draw grid, handle events, run/animate episodes, plotting, tables, CSV export.
- gridworld_app.py
  - entrypoint: constructs Grid and runs GridWorldGUI.
- tests/test_gridworld_logic.py
  - basic tests for environment and Q-learning update.
- requirements.txt
  - list: matplotlib, pillow (PIL), pytest (optional)
- README.md
  - short usage + run instructions.

Implementation details & important gotchas
- PhotoImage lifetime: when using PIL ImageTk.PhotoImage, always pass a `master` parameter (e.g., the Tk root) or store the PhotoImage on a persistent attribute to avoid Tcl GC `pyimage` errors.
- Canvas scaling: bind `<Configure>` and redraw grid cells relative to canvas width/height so resizing works.
- Reachability: when toggling a blocked cell, check `grid.is_reachable()` using BFS; disallow changes that block all paths from `start` to `target`.
- Monte Carlo: use first-visit returns; the agent should store `returns_sum[s]` and `returns_count[s]` and compute `V[s] = sum/count`. Trainer must pass correct `episode_states` and `rewards` lengths (states length = rewards length + 1).
- Single-step accumulation: maintain `_ongoing_states`, `_ongoing_rewards`, `_ongoing_transitions` in the GUI; initialize when first single-step pressed; append each step; when Q-learning, call `update` per step; when `done` occurs, if Monte Carlo use `process_episode` with the accumulated sequences, then clear buffers.
- Save CSV: create `results_csv` directory inside the package with `os.makedirs(..., exist_ok=True)` and write rows (ep, step, s_x, s_y, action, reward, next_s_x, next_s_y, done).
- Trainer.train stop mechanism: check `_stop_requested` each episode to allow graceful stopping between episodes.

Defaults for quick usage
- Grid M=5, N=3, default blocked [(2,2),(2,1)] if within bounds.
- start=(0,2), target=(M-1,N-1)
- max_steps default 20 (but Monte Carlo training should use longer episodes and higher exploration; GUI may auto-increase max_steps/epsilon when policy=mc during a long train run)
- reward scheme: 0 at goal else -1

Developer/run instructions
- Python 3.10+ recommended.
- requirements.txt: matplotlib, pillow
- Run tests: `pytest -q` (or run tests programmatically if path issues exist).
- Run GUI: `python gridworld_app.py`

Deliverables and acceptance criteria
- Project runs and GUI appears without Tcl image errors.
- Q-learning and Monte Carlo agents learn (with appropriate hyperparameters and episode length); Monte Carlo uses state-value returns.
- Single-step accumulation: repeated single-step presses produce a sequence equivalent to an episode (until `done`), with Q-updates for Q-learning and final MC updates for MonteCarlo.
- Value and Q tables accessible and correct.
- CSVs saved in `results_csv` folder.

If you (the operator) want me to implement this from scratch, follow these steps:
1) Create the files above and implement classes and GUI as specified.
2) Run unit tests and fix any issues iteratively.
3) Manually exercise GUI: try single-step accumulation, single-episode run, training runs for both policies.

Notes for reproducer
- Keep code modular: logic in `gridworld_logic.py` should be independent of GUI except for the `Grid` object passed around.
- Provide clear docstrings for `Trainer.run_episode` and `MonteCarloAgent.process_episode` to avoid off-by-one errors.

---

End of prompt file.
