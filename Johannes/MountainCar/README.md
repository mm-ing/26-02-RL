# MountainCar Reinforcement Learning

This project implements a MountainCar agent trainer and Tkinter GUI using Gymnasium and PyTorch.

## Implemented policies

- Dueling DQN (`DuelingDQN`)
- D3QN (Double + Dueling DQN, class `D3QN`)
- DDQN + Prioritized Experience Replay (class `DDQN_PER`, GUI label `DDQN+PER`)

## File structure

- `MountainCar_app.py` - app entry point
- `MountainCar_logic.py` - environment, policies, replay buffers, trainer
- `MountainCar_gui.py` - full GUI, training thread, live plotting, save actions
- `tests/test_MountainCar_logic.py` - pytest unit tests
- `requirements.txt` - dependencies
- `results_csv/` - sampled transitions output
- `plots/` - saved plot PNGs

## Environment configuration

The GUI and logic support these MountainCar setup values:

- `goal_velocity`
- `x_init`
- `y_init`

If an option is not accepted by the local Gymnasium build, the logic falls back gracefully to standard reset behavior.

## Run

From this folder:

```bash
python MountainCar_app.py
```

## Tests

```bash
pytest -q
```

## GPU usage

PyTorch automatically uses CUDA when available; otherwise it runs on CPU.

## Output locations

- Transition CSVs: `results_csv/`
- Plot PNGs: `plots/`
