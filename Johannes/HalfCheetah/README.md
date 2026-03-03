# HalfCheetah (SB3)

This project provides a Tkinter GUI and a reusable logic layer for training and evaluating `HalfCheetah-v5` agents with Stable-Baselines3.

## Exposed Policies
- `PPO`
- `SAC`
- `TD3`

## Environment
- Gymnasium environment ID: `HalfCheetah-v5`
- Runtime configurable environment parameters:
  - `forward_reward_weight`
  - `ctrl_cost_weight`
  - `reset_noise_scale`
  - `exclude_current_positions_from_observation`

## Files
- `HalfCheetah_app.py`: app entrypoint and startup env guards
- `HalfCheetah_logic.py`: environment wrapper, SB3 policy wrapper, trainer
- `HalfCheetah_gui.py`: GUI layout, controls, queue bridge, plotting
- `tests/test_HalfCheetah_logic.py`
- `tests/test_HalfCheetah_gui.py`
- `results_csv/` and `plots/` outputs

## Install
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python HalfCheetah_app.py
```

## Tests
Use isolated tests from this directory:

```bash
python -m pytest -q --rootdir . --confcutdir . tests
```
