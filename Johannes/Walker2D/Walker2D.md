# Walker 2D (Project-specific file)

## Project name
- `Walker2D`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.

## Environment details
- Environment ID: `Walker2d-v5` (gymnasium)
- Runtime-configurable environment parameters:
  - `forward_reward_weight` (default 1)
  - `ctrl_cost_weight` (default 1e-3)
  - `forward_reward_weight` (default 1)
  - `healthy_reward` (default 1)
  - `terminate_when_unhealthy` (default True)
  - `healthy_z_range` (default (0.8, 2))
  - `healthy_angle_range` (default (-1, 1))
  - `reset_noise_scale` (default 5e-3)
  - `exclude_current_positions_from_observation` (default True)
- !!!Render mode for GUI environment: `rgb_array`!!!

## Policies to expose
- `PPO`
- `SAC`
- `TD3`

## GUI
- in the environment group: 
    - add toggles or input fields for each environment parameter below the update button
    - align them in two columns
- in the specific group: 
    - add specific input panels for each policy



