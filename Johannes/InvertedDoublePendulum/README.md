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
- Device is selectable as `CPU` / `GPU` (default `CPU`); if CUDA is unavailable, selection safely falls back to `CPU`.
- Compare mode uses bounded parallelism (`max 4`).
- Compare runs use unique internal run IDs per Cartesian combination so parallel runs never overwrite each other in live plot/history.
- Runtime animation hotfix is included: disabling `Animation on` immediately clears queued replay playback and updates render status.
- Shutdown robustness fix is included: on reset/window-close, paused workers are resumed before stop so the app process exits cleanly.

## Current default baseline (GUI)
- General: `Max steps=1000`, `Episodes=100`, `Gamma=0.99`, epsilon defaults set for deterministic SB3 rollouts.
- PPO-specific defaults: `Hidden layer=256`, `Activation=Tanh`, `LR=3e-4`, `Batch size=128`, `Replay size=100000`, `Learning start=0`.
- SAC/TD3 policy defaults: larger replay buffer and warmup (`Replay size=500000`, `Learning start=10000`) for improved off-policy stability.
