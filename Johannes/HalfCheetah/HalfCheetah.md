# HalfCheetah (Project-specific file)

## Project name
- `HalfCheetah`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.
- Exposed policies for this project are `PPO`, `SAC`, and `TD3`.

## Environment details
- Environment ID: `HalfCheetah-v5` (gymnasium)
- Runtime-configurable environment parameters:
  - `forward_reward_weight` (default 1)
  - `ctrl_cost_weight` (default 0.1)
  - `reset_noise_scale` (default 0.1)
  - `exclude_current_positions_from_observation` (default True)
- Render mode for GUI environment: `rgb_array`

## GUI
- in the environment group: add toggles or input fields for each environment parameter below the update button
- in the specific group: add specific input panels for each policy if applicable

## Policies to expose
- `PPO`
- `SAC`
- `TD3`

