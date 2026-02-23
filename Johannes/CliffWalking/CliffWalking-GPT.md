Implement CliffWalking RL project (Gymnasium + DQN + Double DQN)
=================================================================

Goal
----
Implement a complete CliffWalking reinforcement learning project in this directory by combining requirements from `RL_initial.md` and `CliffWalking.md`.

Create these files
------------------
- `cliffwalking_app.py`
- `cliffwalking_logic.py`
- `cliffwalking_gui.py`
- `tests/test_cliffwalking_logic.py`
- `requirements.txt`
- output folders: `results_csv/`, `plots/`

Implementation contract
-----------------------
Follow object-oriented design and keep behavior aligned with the two spec files.

1) Environment and animation
- Use `gym.make('CliffWalking-v1')` as the environment foundation.
- Do not implement your own Grid!!!! use gymnasium
- Implement a dedicated environment wrapper/class in `cliffwalking_logic.py`.
- Include optional slippery cliff behavior (default: disabled): with configurable probability, apply a perpendicular action instead of intended action.
- Keep standard CliffWalking semantics:
   - start at bottom-left, goal at bottom-right,
   - cliff along the bottom row between start and goal,
   - stepping into cliff causes large negative reward and reset to start.
- Provide data needed by GUI rendering and step-by-step animation.
- Use pygame for animation rendering inside the environment visualization workflow (integrated with the Tkinter app lifecycle).

2) Policies to implement
- `DQNetwork`: vanilla deep Q-learning policy.
- `DDQNetwork`: double DQN policy.
- Suggest and implement sensible defaults for:
   - learning rate,
   - gamma,
   - epsilon schedule,
   - replay buffer size,
   - batch size,
   - target network sync interval,
   - hidden layer size and activation.
- Include replay buffer and transition structure required for DQN training.

3) Trainer
- Expand `Trainer` to train both `DQNetwork` and `DDQNetwork`.
- Implement:
   - `run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)`
   - `train(policy, num_episodes, max_steps, epsilon, save_csv=None)`
- `progress_callback(step)` must be called each step when provided.
- `train(...)` must optionally save sampled transitions to CSV in `results_csv/`.

4) GUI layout and behavior (`cliffwalking_gui.py`)
- Use Tkinter with labeled panels and match `RL_initial.md` structure.
- Apply project-specific overrides from `CliffWalking.md`:
   - Environment panel:
      - Do not implement your own Grid!!!! use gymnasium
      - show map and moving player,
      - animate during training,
      - add toggle `slippery cliff`.
   - Controls panel:
      - remove `Run single step`,
      - keep relevant run/reset/save controls,
      - add `Clear plots` button to clear live plot window.
   - Add `DNN Parameters` panel between `Controls` and `Training Parameters`, tight width and same height behavior as specified.
   - Add relevant input fields across `Training Parameters` and `DNN Parameters` (policy choice, episodes, max steps, epsilon-related settings, DQN/DDQN hyperparameters).
   - Live plot:
      - plot episode rewards,
      - add moving average reward line with double line width.
- Keep GUI responsive:
   - training on background thread,
   - UI updates via `after(0, ...)`,
   - throttle plot updates around 150ms.

5) Tests
- Add `tests/test_cliffwalking_logic.py` covering at least:
   - environment step/cliff/terminal behavior,
   - `Trainer.run_episode` for both `DQNetwork` and `DDQNetwork`,
   - one basic learning-update sanity check per policy path.
- Use `pytest` and ensure tests pass.

6) Dependencies
- `requirements.txt` must include minimum runtime/test dependencies:
   - gymnasium
   - pygame
   - matplotlib
   - pillow
   - numpy
   - pytest
   - torch

Execution steps (assistant must run)
------------------------------------
1. Implement all required files.
2. Install dependencies.
3. Run `pytest -q` and fix failures.
4. Start app with `python cliffwalking_app.py` for smoke check.

Acceptance criteria
-------------------
- Only two policy options: vanilla DQN and double DQN.
- GUI includes `slippery cliff` toggle and `Clear plots` button.
- `DNN Parameters` panel exists between `Controls` and `Training Parameters`.
- Live plot includes moving average with doubled linewidth.
- Training runs asynchronously and updates state/plot during execution.
- Tests pass.

Output format expected from assistant after implementation
---------------------------------------------------------
- Short summary of created/updated files.
- Test command and result.
- Any assumptions/default hyperparameters chosen.
