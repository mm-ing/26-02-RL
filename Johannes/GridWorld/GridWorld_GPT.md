Use this prompt to instruct the assistant to implement the GridWorld labyrint RL project described in the provided markdown:

Task: Implement a complete, runnable GridWorld reinforcement-learning project (Python) with GUI, logic, algorithms, tests, and docs.

Requirements (must implement exactly):

- Project structure and files:
  - Create these files: `gridworld_app.py`, `gridworld_logic.py`, `gridworld_gui.py`, plus `requirements.txt`, `README.md`, and tests (e.g., `tests/test_gridworld_logic.py`).
  - Use object-oriented design: clear classes for Map/Grid, Agent, MonteCarloPolicy, QLearningPolicy, Trainer, and any helpers.

- Grid / environment:
  - Configurable grid size: columns M, rows N (defaults: N=3, M=5).
  - Represent empty fields (white), blocked fields (grey). Default blocked fields: (2,0) and (2,1) (x,y).
  - Blocked fields cannot be entered; ensure at least one valid path always exists from start to target.
  - Default start position: (0,2). Default target: bottom-right cell (M-1, N-1).
  - State s is agent position as (x, y) tuple of ints.
  - Actions: up=0, down=1, left=2, right=3 (no diagonals).
  - Reward: 0 if position == target, else -1.
  - Transition model: implement deterministic transitions by default, but support a configurable stochastic noise parameter (probability to slip to a random neighboring allowed cell). Document default behavior.

- Policies & learning:
  - Implement Monte Carlo policy (first-visit or every-visit—choose and document) and Q-learning.
  - Configurable hyperparameters: learning rate `alpha`, discount factor `gamma`, `max_steps_per_episode`, `num_episodes`.
  - Save sampled transitions to CSV if requested; filename derived from key params (include timestamp).
  - Provide functions to return / display value table and Q-table.

- GUI (`tkinter`):
  - Layout: left section for grid map (map on top), right section for learning controls and plot.
  - Grid features:
    - Display agent as blue circle, target as green rounded square, blocked fields grey.
    - Inputs: grid size, blocked fields (editable by clicking on grid), start pos (drag & drop), target pos (drag & drop).
  - Learning controls:
    - Input fields for `gamma`, `alpha`, `max_steps_per_episode`, `num_episodes`.
    - Buttons: select policy (Monte Carlo or Q-learning), run single step, run single episode, train-and-run, live reward plot toggle, show value table, show Q-table, save samplings CSV.
    - Tooltips for each button/input with short explanations.
  - Plotting: show live reward/episode plot using `matplotlib` embedded in tkinter.

- Testing & docs:
  - Add unit tests for core logic in `gridworld_logic.py` (transition, reward, blocked-field enforcement, Monte Carlo & Q-learning update steps).
  - Provide `requirements.txt` with exact packages used.
  - Add `README.md` with quick setup and example commands to run the GUI, run tests, and run training headless.

Deliverables & Acceptance criteria:
- All files listed above implemented and runnable on Windows with the given `requirements.txt`.
- Tests run and pass (`pytest`).
- GUI opens and supports editing grid, placing agent/target, starting training, and plotting reward.
- Code is reasonably documented and follows the specified defaults and behaviors.

Implementation notes / constraints for the assistant:
- Prefer simple, readable code; avoid unnecessary complexity.
- Use deterministic transitions by default but include optional stochasticity parameter.
- For Monte Carlo, pick either first-visit or every-visit—state choice in README.
- Keep GUI responsive during training (use threads or `after` calls).
- Save CSV with headers: episode, step, s_x, s_y, action, reward, next_s_x, next_s_y, done.

Output requested from you (assistant):
- Create the files listed and implement the project.
- Run the tests and report results.
- If any design choices are made (e.g., first-visit vs every-visit MC), state them concisely.
- Provide short instructions to run the app and tests.

If you need clarifications, ask one question at a time.
