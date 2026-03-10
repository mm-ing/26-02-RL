# Pusher

SB3-based RL workbench for `Pusher-v5` with a Tkinter GUI and matplotlib live plotting.

## Included files
- `Pusher_app.py`
- `Pusher_logic.py`
- `Pusher_gui.py`
- `Pusher_REQUIREMENTS_MATRIX.md`
- `tests/test_Pusher_logic.py`
- `tests/test_Pusher_gui.py`
- `requirements.txt`
- `pytest.ini`
- output folders: `results_csv/`, `plots/`

## Features
- Backend: Stable-Baselines3 (`PPO`, `SAC`, `TD3`, `DDPG`)
- Environment: `Pusher-v5` with runtime-adjustable parameters:
  - `reward_near_weight`
  - `reward_dist_weight`
  - `reward_control_weight`
- Device selection: `CPU` and `GPU` (safe fallback to CPU when CUDA is unavailable)
- Worker-threaded training with pause/resume/cancel controls
- Event bridge contract for GUI updates:
  - `episode`
  - `training_done`
  - `error`
- Deterministic evaluation checkpoints every 10th episode
- Optional sampled transition CSV export to `results_csv/`
- Plot snapshot export to `plots/`
- Non-blocking animation playback with latest-buffer-wins behavior

## Run
```bash
python Pusher_app.py
```

## Test
```bash
python -m pytest -q --rootdir . --confcutdir . tests
```

## Notes
- GUI training environment uses `render_mode="rgb_array"`.
- The logic layer is UI-independent and can be used in headless scripts/tests.
