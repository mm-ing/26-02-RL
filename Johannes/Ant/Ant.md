# Ant (Project-specific file)

## Project name
- `Ant`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.
- Use `EvoTorch` as evolutionary backend.

## Environment details
- Environment ID: `Ant-v5` (gymnasium)
- Runtime-configurable environment parameters:
  - `forward_reward_weight` (default 1)
  - `ctrl_cost_weight` (default 0.5)
  - `contact_cost_weight` (default 5e-4)
- !!!Render mode for GUI environment: `rgb_array`!!!

## Policies to expose
- reinforcement learning:
    - `TQC`
    - `SAC` (default)
- evolutionary learning:
    - `CMA-ES`

## GUI
- in the environment group: 
    - add toggles or input fields for each environment parameter below the update button
    - align them in two columns
    - use Reacher-style paired layout (`label,input | label,input`) for consistent visual behavior
- in the specific group: 
    - add specific input panels for each policy
    - use Reacher-style paired layout (`label,input | label,input`) for shared and policy-specific rows

## Default parameters
- define default parameters for each policy optimized for this approach
- default number of episodes: 3000
for 


