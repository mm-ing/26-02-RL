# Car Racing (Project-specific file)

## Project name
- `CarRacing`

## Backend
- Use `Stable-Baselines3 (SB3)` as RL backend.

## Environment details
- Environment ID: `CarRacing-v3` (gymnasium)
- Runtime-configurable environment parameters:
  - `lap_complete_percent` (default 0.95)
  - `domain_randomize` (default False)
  - `continuous` (default True)
- !!!Render mode for GUI environment: `rgb_array`!!!

## Policies to expose
- continous:
    - `PPO`
    - `SAC` (default)
    - `TD3`
- discrete:
    - double DQN (`DDQN`)
    - `QR-DQN`
- use CNN architecture
- environment continous mode should switch automatically when choosing discrete or continous policies

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


