RL Project Template — initial notes

Purpose
- Reusable, concise template capturing structural, API and UX choices for a small tabular Reinforcement Learning project (GridWorld-style). Use this as the starting README / prompt for new RL tasks.

1. File layout
- gridworld_logic.py  — core environment and algorithms
  - Grid: step(s, action, noise=0.0) -> (s2, reward, done); valid(s); is_reachable(start,target)
  - QLearningAgent: get_q(s,a), best_action(s), update(s,a,r,s2)
  - MonteCarloAgent (state-value): process_episode(episode_states, rewards), get_v(s), best_action(s)
  - Trainer: run_episode(policy, epsilon, max_steps) -> (transitions_or_states, total_reward); train(..., save_csv=None)
- gridworld_gui.py    — Tkinter GUI, drawing, controls, plotting, CSV export, animations
- gridworld_app.py    — simple entrypoint to launch GUI
- tests/test_gridworld_logic.py — unit tests (headless) for Grid and Q-learning
- requirements.txt, README.md, results_csv/ (output)

2. Environment API (recommendations)
- Coordinates: (x,y) with x in [0..M-1] left->right, y in [0..N-1] top->bottom
- step contract: deterministic intended move, optional slip via noise param, return (next_state, reward, done)
- Reward convention: specify clearly (e.g., 0 at goal, -1 per non-terminal step) — treat this as a tunable design choice

3. Agent interfaces
- Q-learning: tabular Q(s,a), TD update, immediate per-step `update` called from Trainer
- Monte Carlo (state-value): first-visit returns; Trainer collects episode_states (len T+1) and rewards (len T) and calls `process_episode(episode_states,rewards)` after episode completes
- Policy usage: Trainer should use epsilon-greedy when `best_action` exists

4. Trainer responsibilities
- Episode execution and optional per-step updates
- Return format: transitions list of (s,a,r,s2,done) or episode states+rewards depending on consumer
- CSV export: create `results_csv` inside package, write header (ep,step,s_x,s_y,action,reward,next_s_x,next_s_y,done)
- Stopping: support a stop flag checked between episodes for graceful interrupts when training in background threads

5. GUI & UX patterns
- Canvas grid that rescales on `<Configure>`; draw cells, agent, target, blocked cells, and path polyline with arrowheads
- Controls: grid size, start/target coords, blocked add/remove, policy select, alpha/gamma/epsilon/max_steps/episodes, live plot toggle
- Actions: `Apply grid`, `Reset map`, `Run single step`, `Run single episode`, `Train and Run`, `Reset All`, `Show Value Table`, `Show Q-table`, `Save samplings CSV`
- Single-step accumulation: maintain `_ongoing_states`, `_ongoing_rewards`, `_ongoing_transitions` that persist across `Run single step` presses until episode terminates or user resets; Q-updates applied per step, MC updated at episode end
- Drag/drop agent & target; hover preview; click-to-toggle blocked cells with BFS reachability check to avoid making target unreachable

6. Plotting and run history
- Embedded matplotlib in Tkinter via `FigureCanvasTkAgg`
- Maintain `plot_runs` history with colors and labels; legend entries clickable to toggle visibility; include horizontal zero-line in ticks

7. Concurrency
- Run long training loops in a background thread; update GUI only via `after()` callback
- Use a shared `_stop_requested` boolean to cancel training between episodes

8. Testing
- Keep logic unit-testable without GUI (avoid importing tkinter in tests)
- Add tests for: `Grid.step`, blocked cell enforcement/reachability, `QLearningAgent.update`

9. Gotchas & implementation notes
- PIL/Tk: keep PhotoImage referenced or pass `master` explicitly to avoid garbage-collected `pyimage` Tcl errors
- Episode indexing: clearly document that `len(episode_states) == len(rewards) + 1`
- When toggling blocked cells, validate `is_reachable()` before committing change
- Provide default hyperparameters and recommend that Monte Carlo training uses longer episodes and higher exploration

10. Reusable prompt snippet (for new projects)
- "Implement a small tabular GridWorld environment with a `Grid` class providing `step(s, action, noise=0.0)` and reachability checks; implement `QLearningAgent` and a state-value `MonteCarloAgent` (first-visit) plus a `Trainer` to run episodes and optionally export CSV. Provide a Tkinter GUI with a resizable canvas showing the grid, controls for hyperparameters, run/training buttons, run-history plotting (matplotlib), and value/Q-table viewers. Keep logic and GUI separate; include headless unit tests for the logic module. Document PhotoImage lifetime mitigation and the episode/state indexing conventions." 

11. Quick checklist for reviewers
- [ ] `Grid.step` returns expected (s2,r,done) for deterministic and noisy moves
- [ ] MonteCarlo `process_episode` uses first-visit returns and correct indexing
- [ ] Single-step accumulation works across sequential presses and MC receives whole episode after terminal step
- [ ] GUI uses `after()` for all background updates and `Reset All` stops training cleanly
- [ ] CSV files written to `results_csv/` with correct columns

End of RL initial template.
