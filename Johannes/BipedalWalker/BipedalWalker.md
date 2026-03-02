# Bipedal Walker (Project-specific file)

## Project name
- `BipedalWalker`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.
- Use `sb3-contrib` for TRPO.

## Environment details
- Environment ID: `BipedalWalker-v3` (gymnasium)
- Runtime-configurable environment parameters:
  - `hardcore` (default `False`)
- Render mode for GUI environment: `rgb_array`

## GUI
- add toggles or input fields for each environment parameter below the update button in the environment group

## Policies to expose
- `PPO`
- `A2C`
- `SAC`
- `TD3`
