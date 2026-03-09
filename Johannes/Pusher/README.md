# Pusher

SB3-based reinforcement learning GUI project for `Pusher-v5` with runtime environment controls and policy-specific training panels.

## Included files
- `Pusher_app.py`: application entrypoint
- `Pusher_gui.py`: Tkinter GUI and worker orchestration
- `Pusher_logic.py`: environment, policy/model builder, trainer, evaluation, event bridge
- `Pusher_REQUIREMENTS_MATRIX.md`: implementation coverage and final contract recheck
- `tests/test_Pusher_logic.py`
- `tests/test_Pusher_gui.py`
- `pytest.ini`, `run_tests.py`
- output folders: `results_csv/`, `plots/`

## Policies
Continuous policies exposed in GUI:
- `PPO`
- `SAC` (default)
- `TD3`
- `DDPG`

## Environment
- ID: `Pusher-v5`
- Render mode for GUI worker env: `rgb_array`
- Runtime-configurable parameters:
  - `reward_near_weight`
  - `reward_dist_weight`
  - `reward_control_weight`

## Run
```bash
python Pusher_app.py
```

## Compare Mode
- Enable `Compare mode` in the `Run` panel.
- Use `Compare grid` to define Cartesian runs with `;`-separated keys and comma-separated values.
- Example:
  - `policy=PPO,SAC,TD3;learning_rate=0.0003,0.0001;batch_size=128,256`
- Compare workers are bounded to `4` concurrent runs.
- In compare mode, animation is enabled for one selected render run and disabled for the other workers.

## Plot PNG Export
- On each completed run (`training_done` event), the current training plot is exported to `plots/`.
- Filenames include policy and key parameters plus timestamp.

## Tests
```bash
python -m pytest -q --rootdir . --confcutdir . tests
```
