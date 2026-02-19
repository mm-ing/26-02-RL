# Gridworld Labyrinth RL

A modular reinforcement learning project with a configurable deterministic Gridworld, tkinter GUI, Monte Carlo and Q-learning, live reward plotting, CSV trajectory export, and animated agent movement.

## Files

- `gridworld_app.py` – application entry point
- `gridworld_logic.py` – environment, policies, scheduler, trajectory/sampling, step-wise API
- `gridworld_gui.py` – tkinter GUI with animation (`after()` scheduling)
- `tests/test_gridworld_logic.py` – unit tests
- `requirements.txt` – Python dependencies

## RL Model

- **State:** `(x, y)`
- **Actions:** `0=up, 1=down, 2=left, 3=right`
- **Transition:** deterministic, blocked/out-of-bounds => stay in place
- **Reward:** `0` at target, else `-1`

## Step-wise API (for GUI animation)

`GridWorldLab` exposes:

- `reset_episode()` -> initial state
- `step()` -> `(prev_state, action, next_state, reward, done)`
- `finish_episode()` -> Monte Carlo end-of-episode update

Q-learning updates are applied online in `step()`.

## Features

- Dynamic grid size and positions
- Click-to-toggle blocked cells
- Drag & drop start/target markers
- Policy selection (Monte Carlo / Q-learning)
- Run single step with immediate redraw
- Run single episode with per-step animation
- Train & run with non-blocking `after()` scheduling
- Stop training gracefully
- Live matplotlib reward plot (`draw_idle` updates)
- Value-table and Q-table popups
- CSV export for sampled transitions

## Install

```bash
cd 26-02-RL/Atefeh/Grid
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 gridworld_app.py
```

## Run tests

```bash
pytest -q
```

## Notes on implementation decisions

- Monte Carlo uses **first-visit** returns to estimate `V(s)`.
- Q-learning uses tabular TD control: `Q <- Q + alpha * (target - Q)`.
- Epsilon is controlled by an exponential decay scheduler with lower bound.
- GUI execution uses `after()` instead of long loops in callbacks, so tkinter can continue event processing and animation rendering.
