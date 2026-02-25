# Taxi Reinforcement Learning Project

This project implements a Gymnasium `Taxi-v3` reinforcement learning app with a Tkinter GUI, PyTorch-based deep Q-learning agents, and test coverage.

## Architecture
- `Taxi_logic.py`: environment wrapper, replay buffers, DQN variants, and trainer.
- `Taxi_gui.py`: full GUI layout (Environment, Controls, Current State, DNN Parameters, Training Parameters, Live Plot) with threaded training and live updates.
- `Taxi_app.py`: entrypoint wiring environment, trainer, and GUI.
- `tests/test_taxi_logic.py`: focused logic tests using `pytest`.

## Policies
- `DQN`: vanilla deep Q-network with target network.
- `DoubleDQN`: double Q target computation.
- `DuelingDQN`: dueling network architecture (value + advantage streams).
- `PrioDQN`: prioritized replay sampling.

## Environment toggles
- `raining`: deterministic movement perturbation for movement actions.
- `running man`: deterministic reward shaping for passenger behavior.

## GUI behavior
- Controls:
  - `Reset All`, `Clear Plot`
  - `Run single episode`, `Save samplings CSV`
  - `Train and Run`, `Save Plot PNG`
- `Current State` shows padded `Training`/`Idle` with step and episode counters.
- `Training Parameters` include episodes, max steps, policy, live plot, reduced speed, MA window (`MA N values`), and `Render every N steps` for animation refresh frequency during training.
- `DNN Parameters` include learning rate, gamma, batch size, buffer size, target update, hidden size, epsilon settings, and prioritized alpha.
- Live plot shows reward and moving average (double linewidth), legend outside right, and visibility checkboxes for each run.

## Run
From this directory:

```bash
python Taxi_app.py
```

## Tests
From this directory:

```bash
pytest tests/test_taxi_logic.py -q
```

## Outputs
- Transition CSVs are written to `results_csv/`.
- Plot PNGs are written to `plots/`.
