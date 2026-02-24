# CliffWalking (DQN / DDQN)

Reinforcement-learning project for Gymnasium `CliffWalking-v1` with a Tkinter GUI and two deep RL policies:
- Vanilla DQN (`DQNetwork`)
- Double DQN (`DDQNetwork`)

The GUI includes:
- Environment animation panel (Gymnasium RGB frames)
- `slippery cliff` toggle
- `Controls`, `Current State`, `DNN Parameters`, `Training Parameters`, and `Live Plot` panels
- Live reward plot with moving average (double line width)

## Project Files

- `cliffwalking_app.py` – app entrypoint
- `cliffwalking_gui.py` – Tkinter GUI and threading/plot behavior
- `cliffwalking_logic.py` – environment wrapper, replay buffer, DQN/DDQN, trainer
- `tests/test_cliffwalking_logic.py` – unit tests
- `requirements.txt` – dependencies
- `results_csv/` – CSV outputs
- `plots/` – saved PNG plots

## Setup

From this directory, install dependencies:

```powershell
pip install -r requirements.txt
```

### GPU setup (recommended when CUDA-capable NVIDIA GPU is available)

`requirements.txt` installs `torch`, but environments sometimes resolve to CPU-only builds.
If `python -c "import torch; print(torch.cuda.is_available())"` returns `False`, reinstall CUDA wheels explicitly:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU visibility:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## Run

```powershell
python cliffwalking_app.py
```

## Tests

Run local CliffWalking tests only:

```powershell
pytest -q tests/test_cliffwalking_logic.py
```

## Notes

- Training runs on a background thread; GUI updates are scheduled on the Tk main thread.
- Plot updates are throttled to reduce UI overhead.
- `Save samplings CSV` writes transitions into `results_csv/`.
- `Save Plot PNG` saves the embedded matplotlib figure into `plots/`.
- `Fast training mode` (in `Training Parameters`) prioritizes throughput by using CPU policy execution and reducing per-step GUI updates during training.

## Recommended default hyperparameters

These are the defaults currently used in the implementation and are a good starting point for both DQN and DDQN:

- Learning rate: `0.001`
- Discount factor (`gamma`): `0.99`
- Epsilon start: `1.0`
- Epsilon end: `0.05`
- Epsilon decay: `0.995`
- Replay buffer size: `10000`
- Batch size: `64`
- Target update frequency: `50` learning steps
- Hidden neurons: `128`
- Activation: `relu`

CliffWalking-specific practical suggestions:
- Keep `slippery cliff` disabled for initial convergence checks.
- Start with `200` episodes and `200` max steps, then scale episodes up if the moving average is still unstable.

## Tuning guide

| Symptom | Likely cause | Try changing |
|---|---|---|
| Rewards stay very low and flat | Exploration remains too random | Lower `epsilon_start` slightly (e.g. `0.8`) or increase `epsilon_decay` toward `0.997` |
| Rewards improve then collapse | Updates too aggressive / unstable targets | Lower learning rate (e.g. `0.0005`) and/or increase target update frequency interval (e.g. `100`) |
| Learning is very slow | Updates too conservative or too noisy | Increase batch size (`64` → `128`) and train more episodes |
| Overreacts to recent transitions | Replay buffer too small | Increase replay buffer size (`10000` → `20000`) |
| DDQN and DQN both plateau early | Capacity too low for policy refinement | Increase hidden neurons (`128` → `256`) |
| Unstable training with slippery cliff on | Environment stochasticity too high for current schedule | Increase episodes, keep `epsilon_end` a bit higher (e.g. `0.1`), reduce learning rate |

## Quick presets

Use these as copy-ready starting points for GUI fields.

### `fast-test` (quick sanity check)

- Policy: `DQN`
- Episodes: `50`
- Max steps: `150`
- Learning rate: `0.001`
- Gamma: `0.99`
- Epsilon start/end/decay: `1.0 / 0.10 / 0.99`
- Replay buffer: `5000`
- Batch size: `32`
- Hidden neurons: `64`
- Target update frequency: `25`
- Activation: `relu`
- Slippery cliff: `off`

### `stable-train` (recommended baseline)

- Policy: `DDQN`
- Episodes: `300`
- Max steps: `200`
- Learning rate: `0.0005`
- Gamma: `0.99`
- Epsilon start/end/decay: `1.0 / 0.05 / 0.995`
- Replay buffer: `10000`
- Batch size: `64`
- Hidden neurons: `128`
- Target update frequency: `50`
- Activation: `relu`
- Slippery cliff: `off`

### `slippery-mode` (stochastic environment)

- Policy: `DDQN`
- Episodes: `500`
- Max steps: `250`
- Learning rate: `0.0003`
- Gamma: `0.99`
- Epsilon start/end/decay: `1.0 / 0.10 / 0.997`
- Replay buffer: `20000`
- Batch size: `128`
- Hidden neurons: `256`
- Target update frequency: `100`
- Activation: `relu`
- Slippery cliff: `on`
