# Humanoid

SB3-based RL workbench for `Humanoid-v5` with a Tkinter GUI and matplotlib live plotting.

## Included files
- `Humanoid_app.py`
- `Humanoid_logic.py`
- `Humanoid_gui.py`
- `Humanoid_REQUIREMENTS_MATRIX.md`
- `Documentation/Documentation.md`
- `tests/test_Humanoid_logic.py`
- `tests/test_Humanoid_gui.py`
- `requirements.txt`
- `pytest.ini`
- folders: `Documentation/`, `benchmarks/`, `results_csv/`, `plots/`

## Features
- Backend: Stable-Baselines3 (`PPO`, `SAC`, `TD3`)
- Environment: `Humanoid-v5` with runtime-adjustable parameters:
  - `forward_reward_weight`
  - `ctrl_cost_weight`
  - `contact_cost_weight`
  - `contact_cost_range`
  - `healthy_reward`
  - `terminate_when_unhealthy`
  - `healthy_z_range`
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
python Humanoid_app.py
```

## Test
```bash
python -m pytest -q --rootdir . --confcutdir . tests
```

## Benchmark

GUI benchmark matrix:

```bash
python benchmarks/benchmark_humanoid_gui.py
```

Headless benchmark:

```bash
python benchmarks/benchmark_humanoid_headless.py
```

Benchmark outputs are written to the `benchmarks/` folder by default.

## Notes
- GUI training environment uses `render_mode="rgb_array"`.
- The logic layer is UI-independent and can be used in headless scripts/tests.
