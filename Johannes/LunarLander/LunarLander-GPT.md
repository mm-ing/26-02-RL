# Executable Prompt: Build LunarLander RL Project

You are GitHub Copilot (GPT-5.3-Codex) working inside this workspace. Implement the LunarLander project in this directory:

- `26-02-RL/Johannes/LunarLander/`

Use the base architecture/rules from:

- `26-02-RL/Johannes/RL_initial_2.0.md`

And apply LunarLander specifics from:

- `26-02-RL/Johannes/LunarLander/LunarLander.md`

## Objective
Create a complete, runnable RL project for Gymnasium Lunar Lander with OOP structure, Tkinter GUI, training thread safety, live plotting, CSV/PNG export, and tests.

## Files to create
Create these files in `26-02-RL/Johannes/LunarLander/`:

- `LunarLander_app.py`
- `LunarLander_logic.py`
- `LunarLander_gui.py`
- `README.md`
- `requirements.txt`
- `tests/test_LunarLander_logic.py`

Create output directories:

- `results_csv/`
- `plots/`

## Core implementation requirements

### Environment
- Use `gym.make('LunarLander-v3', continuous=False, gravity=..., enable_wind=..., wind_power=..., turbulence_power=..., render_mode=...)`.
- Add configurable environment settings with defaults:
  - `gravity=-10.0`
  - `enable_wind=False`
  - `wind_power=15.0`
  - `turbulence_power=1.5`
- Support reset/rebuild when environment parameters are updated from GUI.
- Use pygame-compatible RGB frame rendering from Gymnasium (`render_mode='rgb_array'`) for display in Tkinter.

### Policies / agents
Implement three policy variants with policy-specific defaults and internally separate parameter storage:

1. Double DQN
2. Dueling DQN (`class DuelingDQN`)
3. D3QN (`class D3QN`) = Double + Dueling
4. Double DQN + Prioritized Replay with policy label `DDQN+PER`
   - Use valid Python class name such as `DDQNPER` and map display label to `DDQN+PER`.

Notes:
- Use PyTorch for all networks/learning.
- Use GPU automatically when available (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).
- Support activation function selection: `ReLU`, `Tanh`, `LeakyReLU`, `ELU`.
- Support hidden-layer sizes from comma-separated input (e.g. `256,256`).
- Respect `replay warmup` and `learning cadence` before/for updates.
- Use raw environment reward only (no reward shaping).

### Trainer
Provide trainer API:

- `run_episode(policy, epsilon=0.1, max_steps=1000, progress_callback=None)`
  - Calls `progress_callback(step)` each step when provided.
- `train(policy, num_episodes, max_steps, epsilon, save_csv=None)`
  - Trains over episodes.
  - Optionally writes sampled transitions to CSV under `results_csv/`.

### GUI (`LunarLander_gui.py`)
Implement full Tkinter app layout and behavior exactly per base spec:

- Panels: `Environment`, `Parameters`, `Controls`, `Current Run`, `Live Plot`.
- `Parameters` panel must be scrollable and show scrollbar only when needed.
- Mouse-wheel scroll active only while cursor is inside parameters panel and scrollbar is visible.
- Parameter groups:
  - `Environment`: update button, animation FPS, gravity, wind on, wind power, turbulence
  - `General`: policy dropdown + general training params
  - `Specific`: policy-specific params including activation/hidden layers/warmup/cadence/etc.
  - `Live Plot`: moving average window
- Controls row with equal-width buttons:
  - `Run single episode`
  - `Train and Run`
  - `Pause`/`Run`
  - `Reset All`
  - `Clear Plot`
  - `Save samplings CSV`
  - `Save Plot PNG`
- `Current Run`:
  - Steps progress bar
  - Episodes progress bar
  - Status text format:
    - `Epsilon: <value> | Current x: <value or n/a> | Best x: <value or n/a>`
- Live plot behavior:
  - Plot reward + moving average per run
  - Keep old runs until `Clear Plot`
  - Same color per run pair, thin transparent reward line + bold MA line
  - No title
  - Legend outside right
  - Base legend label format:
    - `<policy> | eps(<epsilon max>/<epsilon min>) | lr=<learning rate>`
  - Exactly two legend entries per run (`... | reward`, `... | MA`)
  - Interactive legend toggles (click legend entry toggles line visibility)
  - Hidden entries visually de-emphasized

### Threading and responsiveness
Must satisfy all of the following:

- Training runs in background thread.
- Worker thread must not update Tk widgets directly.
- Worker thread must not read Tk variables directly; snapshot values on main thread before training starts.
- Use periodic main-thread pump (~20–50 ms) to apply coalesced pending state.
- Worker writes only latest pending state into thread-safe shared structure.
- Serialize env `reset/step/render` access with lock.
- Main thread renders environment frames on configured animation FPS.
- Per-step UI updates must not force full canvas redraw.
- Throttle plot redraws to ~150 ms via timestamp.
- Plot redraw only when new points exist.
- `Reset All` and `Clear Plot` are safe during active training.

### Save/export
- `Save samplings CSV`: produce CSV in `results_csv/`.
- `Save Plot PNG`: save current matplotlib figure in `plots/`.
- PNG filename must encode policy, epsilon min/max, learning rate, gamma, episodes, max_steps, timestamp.

## Tests
Create `tests/test_LunarLander_logic.py` with pytest coverage for:

- env/step behavior
- reachability-style helper if present (or equivalent validity checks)
- `Trainer.run_episode` for different policies
- basic learning updates

Add lightweight GUI smoke checks (if feasible without flakiness) for:

- responsiveness during longer run with live plot
- repeated `Clear Plot` and `Reset All` during training without exceptions
- legend toggling still functional after multiple runs and resets

## Dependencies
`requirements.txt` should include at least:

- `gymnasium[box2d]`
- `pygame`
- `torch`
- `matplotlib`
- `pillow`
- `pytest`
- `numpy`

Do not change the Python environment globally; only update project files.

## Run and verify
After implementation:

1. Run tests with `pytest` from this folder.
2. Fix only issues relevant to this task.
3. Ensure app starts via `LunarLander_app.py`.
4. Update README with setup/run/test usage and key architecture notes.

## Acceptance checklist
- All required files exist.
- App launches and shows required GUI structure.
- Training is threaded and GUI remains responsive.
- Policies include Dueling DQN, D3QN, and DDQN+PER label support.
- Environment parameters (gravity/wind/wind power/turbulence) are editable and applied.
- Live plot behavior + interactive legend meet spec.
- CSV and PNG export work to correct folders.
- Tests pass for implemented logic.
