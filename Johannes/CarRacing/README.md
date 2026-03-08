# CarRacing (SB3)

This directory contains a project-specific CarRacing trainer and GUI using Stable-Baselines3.

## Environment
- Gymnasium environment: `CarRacing-v3`
- Runtime environment parameters:
  - `lap_complete_percent` (default `0.95`)
  - `domain_randomize` (default `False`)
  - `continuous` (switched automatically by selected policy)
- GUI render mode is `rgb_array`

## Exposed Policies
- Continuous: `PPO`, `SAC` (default), `TD3`
- Discrete: `DDQN`, `QR-DQN`

## GUI Notes
- Parameter input controls use Walker2D-style sizing: fixed label column + shared expandable input column for consistent visual field widths.
- In `Current Run`, `Episodes` progress is shown on a row below `Steps` progress.
- In `Specific`, shared parameters are shown first, then a horizontal separator, then policy-specific parameters.
- `activation` is available as a selector (`ReLU`, `Tanh`) and is wired into SB3 `policy_kwargs.activation_fn`.
- `lr_strategy` is a selector (`constant`, `linear`, `exponential`) and drives the logic learning-rate schedule.
- `Live Plot` parameter group is intentionally minimal and keeps only `Moving average values`.
- Replay frame capture amount is controlled by `Frame stride`; no additional advanced capture controls are required.
- Training animation capture is callback-based (captured during `model.learn`) and gated by `Update rate (episodes)`.
- Captured frames are sampled using `Frame stride` and emitted as per-episode playback buffers.
- Playback queue uses latest-wins pending semantics: when playback is active, only the newest pending episode animation is kept.

## Files
- `CarRacing_app.py`: app entry point
- `CarRacing_logic.py`: environment wrapper, SB3 policy factory, trainer
- `CarRacing_gui.py`: Tkinter GUI and threaded training bridge
- `CarRacing_REQUIREMENTS_MATRIX.md`: implementation coverage matrix
- `tests/`: logic + GUI tests

## Quick Start
1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Run GUI:
   - `python CarRacing_app.py`
3. Run tests:
   - `python -m pytest -q --rootdir . --confcutdir . tests`

## Notes
- Default episodes are set to `3000`.
- Sample transitions can be exported to `results_csv/`.
- Plot export writes PNG files to `plots/`.
- Current isolated test status: `16 passed`.
