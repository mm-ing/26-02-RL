# LunarLander 2.0 (Project-specific file)

## Project name
- `LunarLander`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.
- Use `sb3-contrib` for TRPO.

## Environment details
- Environment ID: `LunarLander-v3` (gymnasium)
- Runtime-configurable environment parameters:
  - `continuous` (`False` for discrete policies, `True` for continuous policies)
  - `gravity` (default `-10.0`)
  - `enable_wind` (default `False`)
  - `wind_power` (default `15.0`)
  - `turbulence_power` (default `1.5`)
- Render mode for GUI environment: `rgb_array`

## Policies to expose
- `DuelingDQN`
- `D3QN`
- `DDQN+PER`
- `PPO`
- `A2C`
- `TRPO`
- `SAC`

## Policy/environment mode rule
- Discrete policies: `DuelingDQN`, `D3QN`, `DDQN+PER` -> force `continuous=False`
- Continuous policies: `PPO`, `A2C`, `TRPO`, `SAC` -> force `continuous=True`
