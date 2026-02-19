# GridWorld RL

Quick runnable GridWorld labyrinth reinforcement-learning example.

Files:

- `gridworld_logic.py`: core environment and algorithms (Grid, QLearningAgent, MonteCarloAgent, Trainer)
- `gridworld_gui.py`: tkinter GUI to edit grid and run training
- `gridworld_app.py`: entry point
- `requirements.txt`: required packages
- `tests/test_gridworld_logic.py`: unit tests

Run GUI:

```bash
python -m Johannes.GridWorld.gridworld_app
```

Run tests:

```bash
pytest -q
```

Design choices:

- Monte Carlo implementation uses first-visit averaging for state values.
- Transitions are deterministic by default; a `noise` parameter in `Trainer` enables stochastic slips.
