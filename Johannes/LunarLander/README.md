# LunarLander RL

Reinforcement learning project for Gymnasium `LunarLander-v3` with Tkinter GUI and a Stable-Baselines3 (PyTorch)-based training backend.

## Implemented policies
- `DuelingDQN`
- `D3QN` (Double + Dueling)
- `DDQN+PER` (mapped to SB3 DQN backend)
- `PPO`
- `A2C`
- `TRPO` (via `sb3-contrib` when available; fallback to PPO backend)
- `SAC`

## Project files
- `LunarLander_app.py` – app entry point
- `LunarLander_logic.py` – environment, agents, trainer
- `LunarLander_gui.py` – GUI, background training, live plotting
- `tests/test_LunarLander_logic.py` – logic tests
- `results_csv/` – saved transition CSV files
- `plots/` – saved plot PNG files

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python LunarLander_app.py
```

## Test
```bash
pytest -q
```

## Notes
- The GUI runs training on a worker thread and applies updates on the Tk main thread.
- The environment is configurable via gravity and wind settings in the `Environment` parameter group.
- The live plot supports interactive legend toggling and preserves past runs until `Clear Plot`.
- The logic layer is SB3-backed while preserving the existing GUI-facing `Trainer` API.
- The `Compare` parameter group provides `Compare on` (default off).
- With `Compare on`, `Train and Run` starts parallel runs for configured compare combinations and appends all runs to the live plot.
- In compare mode, environment animation, status text, and progress bars follow the currently selected policy in the policy dropdown.
