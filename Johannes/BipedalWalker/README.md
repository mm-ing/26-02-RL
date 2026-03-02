# BipedalWalker (SB3)

Tkinter-based reinforcement learning UI for `BipedalWalker-v3` using `Stable-Baselines3`.

## Files
- `BipedalWalker_app.py`
- `BipedalWalker_gui.py`
- `BipedalWalker_logic.py`
- `tests/test_BipedalWalker_logic.py`
- `tests/test_BipedalWalker_gui.py`

## Features
- Policies: `PPO`, `A2C`, `SAC`, `TD3`
- Runtime environment parameter: `hardcore`
- Background training with queue-driven UI updates
- Live reward / moving average / evaluation plotting
- Compare mode with cartesian parameter combinations
- CSV export to `results_csv/`
- PNG export to `plots/`

## Run
```bash
python BipedalWalker_app.py
```

## Test
```bash
pytest -q
```

Isolated test helper:
```bash
python run_tests.py
```
