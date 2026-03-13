# Humanoid (Project-specific file)

## Project name
- `Humanoid`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.

## Environment details
- Environment ID: `Humanoid-v5` (gymnasium)
- Runtime-configurable environment parameters:
  - `forward_reward_weight` (default 1.25)
  - `ctrl_cost_weight` (default 0.1)
  - `contact_cost_weight` (default 5e-7)
  - `contact_cost_range` (default (-np.inf, 10.0))
  - `healthy_reward` (default 5.0)
  - `terminate_when_unhealthy` (default True)
  - `healthy_z_range` (default (1.0, 2.0))
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
- default number of episodes: 3000 (or suggest one)

## Documentation
- add doc strings in your code
- generate a `Documentation/Documentation.md` with the following structure:
    - 1. Workbench
        - describe the functionality of the three prompt files `general.md`, `logic.md` and `gui.md` in three subsections
        - add a screenshot of the final GUI, describe its funcionality
    - 2. Comparison of Methods and Parameters
        - 2.1. Environment
            - describe the environment
        - 2.2. Methods
            - describe the methods and start parameters based on `AlgortihmSelection.md`

