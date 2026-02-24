# FrozenLake RL (Gymnasium + DQN variants)

This project implements a FrozenLake reinforcement learning playground with a Tkinter GUI and PyTorch-based deep RL policies.

## Implemented algorithms

- `DQN` (vanilla deep Q-learning)
- `DoubleDQN`
- `DuelingDQN`
- `PrioDQN` (prioritized replay)

## Features

- Gymnasium `FrozenLake-v1` environment
- Configurable environment settings:
  - `is_slippery`
  - `map_name` (`4x4`, `8x8`)
  - `success_rate` for intended action probability (custom transition model)
- GUI panels:
  - `Environment`
  - `Controls`
  - `Current State`
  - `DNN Parameters`
  - `Training Parameters`
  - `Live Plot`
- Threaded training with responsive UI (`after(0, ...)` updates)
- Live reward plot with moving average (`MA N values`) and clickable legend
- Export options:
  - transition samplings CSV to `results_csv/`
  - plot PNG to `plots/`

## File structure

- `frozenlake_app.py` - entrypoint
- `frozenlake_logic.py` - environment, policies, trainer
- `frozenlake_gui.py` - full GUI implementation
- `tests/test_frozenlake_logic.py` - unit tests
- `requirements.txt` - dependencies
- `results_csv/` - generated transition exports
- `plots/` - generated plot images

## Install

```bash
pip install -r requirements.txt
```

## Run GUI

```bash
python frozenlake_app.py
```

## Run tests

```bash
pytest -q
```

## Notes

- If CUDA is available, models use GPU automatically.
- The custom `success_rate` is implemented by overriding FrozenLake transition probabilities.
- Rendering is done via Gymnasium `rgb_array`; Pillow is used to display frames in Tkinter.
