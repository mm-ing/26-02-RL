# Pusher (Project-specific file)

## Project name
- `Pusher`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.

## Environment details
- Environment ID: `Pusher-v5` (gymnasium)
- Runtime-configurable environment parameters:
  - `reward_near_weight` (default 0.5)
  - `reward_dist_weight` (default 1)
  - `reward_control_weight` (default 0.1)
- !!!Render mode for GUI environment: `rgb_array`!!!

## Policies to expose
- continous:
    - `PPO`
    - `SAC` (default)
    - `TD3`
    - `DDPG`

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


