# Executable Prompt: Build the `CartPole` RL Project

You are GitHub Copilot (GPT-5.3-Codex). Implement a complete CartPole reinforcement learning project in this folder:

- `26-02-RL/Johannes/CartPole/`

Follow **all** requirements from `../RL_initial.md` and `CartPole.md`, using object-oriented design and minimal, clean architecture.

## Objective
Create a runnable RL app with:
- Gymnasium CartPole environment
- 3 DQN-based approaches:
  - `DoubleDQN`
  - `DuelingDQN`
  - `D3QN` (Double + Dueling)
- Tkinter GUI with embedded live Matplotlib plots
- Background-thread training with responsive UI updates
- CSV and PNG export support
- Unit tests for core logic

## Constraints
- Do **not** change the active Python environment.
- Python 3.8+ compatible.
- Use `PyTorch` (GPU-aware: use CUDA if available, fallback to CPU).
- Keep implementation focused; do not add unrelated features.

## Required files to create
Create these files in `CartPole/`:
- `CartPole_app.py`
- `CartPole_logic.py`
- `CartPole_gui.py`
- `README.md`
- `requirements.txt`
- `tests/test_CartPole_logic.py`
- directories: `results_csv/`, `plots/`

## Environment & logic requirements (`CartPole_logic.py`)
1. Environment wrapper
   - Build CartPole env with:
     - `gym.make('CartPole-v1', render_mode='rgb_array')` for GUI frame rendering
   - Support toggle parameter:
     - `sutton_barto_reward: bool = False`
   - If `sutton_barto_reward=True`, use modified reward shaping (document exact formula in README and code docstring).

2. Replay buffer and trainer support
   - Add replay buffer for off-policy DQN updates.
   - Implement a `Trainer` class with:
     - `run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)`
       - Calls `progress_callback(step)` every step when provided.
       - Returns episode reward and sampled transitions for optional CSV storage.
     - `train(policy, num_episodes, max_steps, epsilon, save_csv=None)`
       - Runs multiple episodes.
       - Optionally writes sampled transitions to CSV under `results_csv/`.

3. Policies / agents
   - Implement classes:
     - `DoubleDQN`
     - `DuelingDQN`
     - `D3QN`
   - Each should:
     - Use PyTorch networks.
     - Have sensible default hyperparameters (gamma, lr, target update cadence, batch size, epsilon-related parameters).
     - Support action selection (epsilon-greedy), memory storage, and learning update.
   - `DuelingDQN` must use value/advantage heads.
   - `DoubleDQN` target computation must use online net argmax with target net evaluation.
   - `D3QN` combines both mechanisms.

4. Utility behavior
   - Include `is_reachable(...)` and `step(...)` style methods or equivalents needed for testability per base requirements.
   - Keep APIs test-friendly and deterministic when seeds are set.

## GUI requirements (`CartPole_gui.py`)
Implement Tkinter GUI with `ttk.LabelFrame` panels and this layout:

1. `Environment` (top, full width)
   - Display current CartPole frame (rendered image) using Pillow when available.
   - Include toggle control for `sutton_barto_reward` (default False).

2. `Controls` (left column), 2x3 buttons:
   - Row0: `Reset All` | `Clear Plot`
   - Row1: `Run single episode` | `Save samplings CSV`
   - Row2: `Train and Run` | `Save Plot PNG`
   - Buttons use `sticky='ew'`.

3. `Current State` (below Controls)
   - Single-line status label with exact padded formatting:
     - `Training: step:    3  episode:   12`
     - `    Idle: step:    0  episode:    0`
   - Prefix padded to length of `Training`.

4. `DNN Parameters` (new panel)
   - Place between `Controls` and `Training Parameters`.
   - Tight width, same vertical span style as Training Parameters.
   - Add relevant model hyperparameter fields (e.g., learning rate, gamma, batch size, replay size, target update, hidden size).

5. `Training Parameters` (right column)
   - Inputs for: max steps, episodes, policy dropdown, moving average window, epsilon min, epsilon max, epsilon decay, animation refresh rate (every Nth step, default 10).
   - Checkboxes:
     - `Live plot` (default True)
     - `reduced speed` (default True)

6. `Live Plot` (bottom, full width)
   - Embedded Matplotlib figure/canvas.
   - Plot live episode reward + moving average (default 20).
   - Same color for reward + moving average of same run.
   - Live reward: thin, slightly transparent.
   - Moving average: bold.
   - Legend outside right.
   - Legend entry names include policy + relevant params.
   - Legend entries must be interactive toggles: clicking legend line/label shows or hides corresponding line.
   - Do not create separate toggle checkbox widgets for legend visibility.
   - Hidden legend entries should be visually de-emphasized (e.g. reduced alpha).

## Responsiveness & threading
- Training must run in background thread.
- Worker thread must not update Tk widgets directly.
- Use a periodic main-thread UI pump (~20–50 ms) that coalesces and applies latest pending updates.
- Worker should write only latest pending state to thread-safe shared state (e.g., episode/step counters, frame-refresh request, rewards snapshot, finalize-run flag).
- Per-step updates should update `Current State` label directly (no full canvas redraw).
- Throttle plot redraws to about 150ms using timestamp field (e.g., `self._last_plot_update`).
- If `reduced speed` is enabled, sleep ~0.033s between episodes.
- `Reset All` and `Clear Plot` must remain safe during and after active training (no crashes).

## Control behavior
1. `Run single episode`
   - Reset agent/environment to configured start.
   - Set `current_episode=1`, `current_step=0` before run.
   - Execute one `Trainer.run_episode(...)` and animate transitions sequentially.

2. `Train and Run`
   - Run in background thread.
   - For each episode:
       - pass `progress_callback` into `run_episode` to update pending progress state
       - update `current_episode`/`current_step` via coalesced UI pump updates
       - append rewards and update latest rewards snapshot for throttled plotting
   - honor reduced speed toggle.

3. `Reset All`
   - Set `_stop_requested` flag.
   - Clear plot runs and environment view/state.
   - Reset active agent instance.

4. `Save samplings CSV`
   - Produce CSV in `results_csv/` via trainer method.

5. `Save Plot PNG`
   - Save figure to `plots/` with filename encoding:
     - policy, eps min/max, alpha, gamma, episodes, max_steps, timestamp

## Entry point (`CartPole_app.py`)
- Instantiate logic + GUI classes and start Tkinter main loop.

## Tests (`tests/test_CartPole_logic.py`)
Add pytest tests that cover:
- environment `step`
- `is_reachable` (or project-equivalent reachability/sanity function)
- `Trainer.run_episode` with different policies
- basic learning update behavior (weights or Q-values change after training steps)
- GUI robustness smoke checks:
   - long training with `reduced speed=False` and `live_plot=True` remains responsive
   - repeated `Clear Plot` and `Reset All` during training do not raise exceptions
   - interactive legend toggling works after multiple runs and clear/reset cycles

Use small episode/step counts to keep tests fast.

## Dependencies (`requirements.txt`)
Include runtime/dev dependencies needed by implementation:
- `gymnasium`
- `torch`
- `numpy`
- `matplotlib`
- `pillow`
- `pygame`
- `pytest`

## README
Write `README.md` with:
- project overview
- architecture summary
- policy descriptions
- configurable parameters
- run instructions
- test command
- output files (`results_csv/`, `plots/`)
- note on CUDA auto-detection

## Commands to validate
Run after implementation:
1. `pytest -q`
2. Launch app:
   - `python CartPole_app.py`

## Clarification to resolve textual inconsistency
In `CartPole.md`, "show the gymnasium animation of taxi" appears to be a typo. Implement and display **CartPole** animation in the Environment panel.

## Definition of done
- All required files exist.
- GUI layout and behavior match requirements.
- All 3 policies implemented and selectable.
- Threaded training + live counters + throttled plotting work.
- CSV and PNG exports work.
- Tests pass with `pytest`.
- README and requirements are complete.
