# Reacher (Project-specific file)

## Project name
- `Reacher`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.

## Environment details
- Environment ID: `Reacher-v5` (gymnasium)
- Runtime-configurable environment parameters:
  - `reward_dist_weight` (default 1)
  - `reward_control_weight` (default 0.1)
- !!!Render mode for GUI environment: `rgb_array`!!!

## Policies to expose
- continous:
    - `PPO`
    - `SAC` (default)
    - `TD3`

## GUI
- in the environment group: 
    - add toggles or input fields for each environment parameter below the update button
    - align them in two columns
- in the specific group: 
    - add specific input panels for each policy

## Default parameters
- define default parameters for each policy optimized for this approach
- default number of episodes: 3000
for 


