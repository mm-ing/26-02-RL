Implement CliffWalking RL project (executable prompt)
=====================================================

Role
----
You are implementing a complete reinforcement-learning project in this directory using object-oriented Python and PyTorch.

Source specs to merge
---------------------
- `RL_initial.md` (general architecture, trainer/GUI/threading/tests structure)
- `CliffWalking.md` (project-specific behavior)

If any requirement conflicts, prioritize `CliffWalking.md` for project-specific choices while keeping `RL_initial.md` architecture.

Project name
------------
- `CliffWalking`

Create/Update these files
-------------------------
- `cliffwalking_app.py`
- `cliffwalking_logic.py`
- `cliffwalking_gui.py`
- `tests/test_cliffwalking_logic.py`
- `requirements.txt`
- ensure output directories exist: `results_csv/`, `plots/`

Core implementation requirements
--------------------------------
1. Environment
   - Build environment around `gym.make('CliffWalking-v1')`.
   - Add optional `slippery cliff` behavior (default: off) where movement can randomly deviate perpendicular to intended direction.
   - Use pygame-based animation frames from gymnasium render mode and show them in the GUI environment panel.

2. Policies (PyTorch)
   - Implement **vanilla DQN** class `DQNetwork`.
   - Implement **Double DQN** class `DDQNetwork`.
   - Suggest and implement sensible defaults, including:
     - learning rate
     - gamma
     - epsilon start/end and decay
     - replay buffer size
     - batch size
     - target network update frequency
     - hidden layer width(s)
     - activation function
   - Use replay buffer + minibatch updates for both methods.
   - For DDQN, use online network for action selection and target network for target value evaluation.

3. Trainer
   - Implement `Trainer.run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)` and call `progress_callback(step)` every step when provided.
   - Implement `Trainer.train(policy, num_episodes, max_steps, epsilon, save_csv=None)` and optionally save sampled transitions to CSV under `results_csv/`.

GUI requirements
----------------
1. Layout
   - Use Tkinter + ttk label frames consistent with `RL_initial.md`:
     - `Environment` (top, full width)
     - `Controls` (left)
     - `Current State` (below Controls)
     - `DNN Parameters` (between `Controls` and `Training Parameters`, tight width, same height as `Training Parameters`)
     - `Training Parameters` (right, tight width)
     - `Live Plot` (bottom, full width)

2. Controls and state
   - Provide controls to run episode/training, reset all, save samplings CSV, save plot PNG, and clear plot.
   - Keep GUI responsive by running training in a background thread and using `after(0, ...)` for UI updates.
   - Update current step/episode live; avoid full redraws just to update counters.

3. Environment panel
   - Display gymnasium cliffwalking animation in the panel (pygame-rendered frames).
   - Add a `slippery cliff` toggle in GUI.

4. Parameter inputs
   - `Training Parameters` panel: include max steps, episodes, policy selector (`DQN` / `DDQN`), epsilon controls, gamma, learning rate, live plot toggle, reduced speed toggle.
   - `DNN Parameters` panel: include replay buffer size, batch size, activation function, hidden neurons, target update frequency.

5. Live plot
   - Plot episode rewards for runs.
   - Add moving average reward curve with double line width.
   - Throttle plot updates to ~150ms.
   - Keep clickable legend behavior to toggle run visibility.

Testing requirements
--------------------
- Add `pytest` tests in `tests/test_cliffwalking_logic.py` for:
  - environment step/reward/termination behavior,
  - slippery toggle effect (deterministic test using seeded RNG or controlled randomness),
  - `Trainer.run_episode` execution for both `DQNetwork` and `DDQNetwork`,
  - one learning update path for DQN and DDQN (lightweight, fast).

Dependencies (`requirements.txt`)
---------------------------------
Include at least:
- gymnasium
- pygame
- matplotlib
- pillow
- numpy
- torch
- pytest

Execution steps (must run)
--------------------------
1. Install dependencies.
2. Run unit tests and fix issues until they pass.
3. Launch the app to verify GUI starts and animation panel updates.

Completion criteria
-------------------
- Code compiles and tests pass.
- GUI launches, shows environment animation, and exposes all required controls/parameters.
- Training works for both DQN and DDQN and updates live plots (including moving average).
- CSV/PNG export works.

Final output format from assistant
----------------------------------
After implementation, provide:
- brief change summary by file,
- test command and result,
- run command for the GUI.
