# Inverted Double Pendulum (Project-specific file)

## Project name
- `InvDoubPend`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.
- Use `sb3-contrib` for TRPO.

## Environment details
- Environment ID: `InvertedDoublePendulum-v5` (gymnasium)
- Runtime-configurable environment parameters:
  - `healthy_reward` (default 10)
  - `reset_noise_scale` (default 0.1)
- Render mode for GUI environment: `rgb_array`

## GUI
- add toggles or input fields for each environment parameter below the update button in the environment group

## Policies to expose
- `PPO`
- `SAC`
- `TD3`
