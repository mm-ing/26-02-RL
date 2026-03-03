# Inverted Double Pendulum (`InvDoubPend`)

This project provides a Tkinter-based RL workbench for Gymnasium `InvertedDoublePendulum-v5` using **Stable-Baselines3 (SB3)**.

## Files
- `InvDoubPend_app.py` – app entrypoint
- `InvDoubPend_gui.py` – GUI, plotting, threading, compare orchestration
- `InvDoubPend_logic.py` – environment wrapper, SB3 policy/trainer logic
- `tests/` – isolated logic and GUI smoke/regression tests
- `results_csv/` – CSV exports
- `plots/` – PNG exports

## Exposed Policies
- `PPO`
- `SAC`
- `TD3`

## Environment Parameters (runtime update)
- `healthy_reward` (default `10`)
- `reset_noise_scale` (default `0.1`)

GUI render-mode environment uses `rgb_array`.

## Install
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python InvDoubPend_app.py
```

## Tests (isolated)
```bash
python -m pytest -q --rootdir . --confcutdir . tests
```

## Notes
- Device is fixed to `CPU` for all policies.
- Compare mode uses bounded parallelism (`max 4`).
- Runtime animation hotfix is included: disabling `Animation on` immediately clears queued replay playback and updates render status.
