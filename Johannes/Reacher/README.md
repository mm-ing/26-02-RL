# Reacher

SB3-based Reacher (`Reacher-v5`) training project with a Tkinter GUI, compare mode, and live matplotlib plotting.

## Files
- `Reacher_app.py`: startup entrypoint with environment guards
- `Reacher_logic.py`: environment wrapper, SB3 agent wrapper, trainer, event contract
- `Reacher_gui.py`: GUI layout, controls, background workers, event pump, animation, live plot
- `tests/test_Reacher_logic.py`: logic unit/smoke tests
- `tests/test_Reacher_gui.py`: GUI smoke/regression tests
- `Reacher_REQUIREMENTS_MATRIX.md`: contract coverage tracker

## Environment Parameters
- `reward_dist_weight` (default `1.0`)
- `reward_control_weight` (default `0.1`)

## Exposed Policies
- `PPO`
- `SAC` (default)
- `TD3`

## Run
```bash
python Reacher_app.py
```

## Test
```bash
python -m pytest -q --rootdir . --confcutdir . tests
```

## Exports
- CSV transitions: `results_csv/`
- Plot PNG: `plots/`
