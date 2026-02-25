# CartPole Reinforcement Learning Project

This project implements CartPole training with three DQN variants:
- Double DQN (`DoubleDQN`)
- Dueling DQN (`DuelingDQN`)
- D3QN (`D3QN`, Double + Dueling)

The app includes a Tkinter GUI with live environment rendering, parameter controls, background training, and live Matplotlib reward plotting.

## Architecture
- `CartPole_app.py`: application entrypoint
- `CartPole_logic.py`: environment wrapper, replay buffer, DQN agents, trainer
- `CartPole_gui.py`: GUI layout, interactions, threading, plotting, exports
- `tests/test_CartPole_logic.py`: unit tests for logic and trainer behavior
- `results_csv/`: exported sampling CSV files
- `plots/`: exported figure PNG files

## Reward setup
The environment uses `gym.make('CartPole-v1', render_mode='rgb_array')`.

Optional `sutton_barto_reward` behavior:
- non-terminal step reward: `0.0`
- terminal/truncated step reward: `-1.0`

Default mode keeps Gymnasium reward.

## Main configurable parameters
### Training Parameters
- max steps
- episodes
- policy (`DoubleDQN`, `DuelingDQN`, `D3QN`)
- moving average window
- epsilon min/max
- `Live plot` toggle
- `reduced speed` toggle

### DNN Parameters
- learning rate
- gamma
- batch size
- replay size
- target update interval
- hidden size

## CUDA / device
Agents auto-select device:
- CUDA GPU if available
- CPU fallback otherwise

## Run
From this folder:

```bash
python CartPole_app.py
```

## Startup self-check
- The GUI performs a first-frame render during startup.
- If Gymnasium requires a reset before render, the environment wrapper auto-resets once and then renders.
- This avoids `gymnasium.error.ResetNeeded` on first launch.

## Tests
Run:

```bash
pytest -q
```

## Output
- Sampling CSV export via **Save samplings CSV** writes into `results_csv/`
- Plot PNG export via **Save Plot PNG** writes into `plots/`
  with policy/hyperparameter/timestamp encoded filenames
- In the live plot, click legend entries to show/hide their corresponding lines.
