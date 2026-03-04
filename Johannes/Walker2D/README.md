# Walker2D

Walker2D training UI using Stable-Baselines3 with Gymnasium `Walker2d-v5`.

## Files
- `Walker2D_app.py`: App entrypoint
- `Walker2D_gui.py`: Tkinter GUI + plotting + background worker bridge
- `Walker2D_logic.py`: Environment wrapper, SB3 model factory, trainer loop
- `tests/test_Walker2D_logic.py`: Logic smoke tests
- `tests/test_Walker2D_gui.py`: GUI smoke test

## Policies
- PPO
- SAC
- TD3

## Default Training Configuration
- Episodes default: `1000`
- Max steps default: `1000`
- Device default: `CPU`

### Specific Group (shared across NN policies, policy-scoped values)
- `gamma`
- `learning_rate`
- `batch_size`
- `hidden_layer` (single width like `256` or comma-separated architecture like `256,128,64`)
- `lr_strategy` (`constant` / `linear` / `exponential`)
- `min_lr`
- `lr_decay`

### Policy-specific defaults (current)
- `PPO`
	- shared defaults: `gamma=0.99`, `learning_rate=3e-4`, `batch_size=128`, `hidden_layer=256`, `lr_strategy=constant`, `min_lr=1e-5`, `lr_decay=1.0`
	- specific defaults: `n_steps=2048`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`
- `SAC`
	- shared defaults: `gamma=0.99`, `learning_rate=3e-4`, `batch_size=256`, `hidden_layer=256`, `lr_strategy=constant`, `min_lr=1e-5`, `lr_decay=1.0`
	- specific defaults: `buffer_size=500000`, `learning_starts=10000`, `tau=0.005`, `train_freq=1`, `gradient_steps=1`
- `TD3`
	- shared defaults: `gamma=0.99`, `learning_rate=1e-3`, `batch_size=256`, `hidden_layer=256`, `lr_strategy=constant`, `min_lr=1e-5`, `lr_decay=1.0`
	- specific defaults: `buffer_size=500000`, `learning_starts=10000`, `tau=0.005`, `policy_delay=2`, `train_freq=1`, `gradient_steps=1`, `target_policy_noise=0.2`, `target_noise_clip=0.5`

Notes:
- All `Specific` parameters are cached per policy and restored on policy switch.
- LR schedule and hidden-layer settings are applied for all exposed NN-based policies.
- `hidden_layer` mapping to SB3 `policy_kwargs.net_arch`:
	- single value `256` -> `[256, 256]` (backward-compatible default)
	- comma-separated value `256,128,64` -> `[256, 128, 64]`

## Environment parameters (runtime configurable)
- `forward_reward_weight`
- `ctrl_cost_weight`
- `healthy_reward`
- `terminate_when_unhealthy`
- `healthy_z_range`
- `healthy_angle_range`
- `reset_noise_scale`
- `exclude_current_positions_from_observation`

## Setup
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python Walker2D_app.py
```

## Tests
```bash
python -m pytest -q --rootdir . --confcutdir . tests
```

## Outputs
- CSV exports: `results_csv/`
- Plot exports: `plots/`

## Requirements Matrix Workflow (Reuse for Future Projects)

For each new project in this repository, create and maintain:
- `<ProjectName>_REQUIREMENTS_MATRIX.md`

Recommended flow:
1. **Generate matrix at project start**
	- Parse the active prompt contracts (`general.md`, `logic.md`, `gui.md`, project file).
	- Build tables with `Requirement`, `Status`, `Notes` for GUI, logic, and tests.
	- Use status values: `✅`, `🟡`, `❌`.
2. **Update matrix after each implementation pass**
	- Every time behavior/tests change, update the affected matrix rows.
3. **Enforce matrix gate before handoff**
	- `High-impact` section must have no unresolved `❌` items.
	- If any `🟡` remains, include explicit rationale and follow-up action.
4. **Run test validation and record result in PR/handoff**
	- `python -m pytest -q --rootdir . --confcutdir . tests`

Suggested pull-request checklist item:
- "Requirements matrix created/updated and high-impact gaps resolved or justified."
