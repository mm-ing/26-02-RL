# LunarLander Project Specification (Current State)

## Goal
Maintain and evolve the `LunarLander` RL project based on Gymnasium `LunarLander-v3`, following the architecture and GUI baseline from `RL_initial_2.0.md`, with project-specific behavior documented here.

## Implemented policy set
- Discrete (`continuous=False`): `DuelingDQN`, `D3QN`, `DDQN+PER`
- Continuous (`continuous=True`): `PPO`, `A2C`, `TRPO`, `SAC`

## Environment & mode behavior
- Environment is created through `gym.make('LunarLander-v3', ...)`.
- Supported environment parameters:
  - `gravity` (default `-10.0`)
  - `enable_wind` (default `False`)
  - `wind_power` (default `15.0`)
  - `turbulence_power` (default `1.5`)
- Mode enforcement is active:
  - Continuous policies force `continuous=True`
  - Discrete policies force `continuous=False`
  - GUI toggle and environment are synchronized when automatic mode switching is required.

## Current default presets
- `DuelingDQN`: gamma `0.99`, lr `3.00e-04`, replay `100000`, batch `128`, target update `200`, replay warmup `5000`, cadence `2`, hidden `256,256,128`, activation `ReLU`, LR strategy `exponential`, LR decay `0.1`, min lr `1.00e-05`.
- `D3QN`: gamma `0.99`, lr `2.50e-04`, replay `150000`, batch `128`, target update `200`, replay warmup `8000`, cadence `2`, hidden `512,256,128`, activation `ReLU`, LR strategy `exponential`, LR decay `0.1`, min lr `1.00e-05`.
- `DDQN+PER`: gamma `0.99`, lr `2.00e-04`, replay `200000`, batch `128`, target update `200`, replay warmup `10000`, cadence `2`, hidden `512,256,128`, activation `ReLU`, LR strategy `exponential`, LR decay `0.1`, min lr `1.00e-05`.
- `PPO`: gamma `0.99`, lr `1.00e-04`, replay `100000`, batch `128`, target update `200`, replay warmup `5000`, cadence `32`, hidden `256,256`, activation `ReLU`, LR strategy `linear`, LR decay `0.3`, min lr `1.00e-05`, `gae_lambda=0.95`, `ppo_clip_range=0.2`.
- `A2C`: gamma `0.99`, lr `1.50e-04`, replay `100000`, batch `128`, target update `200`, replay warmup `5000`, cadence `16`, hidden `256,256`, activation `ReLU`, LR strategy `exponential`, LR decay `0.3`, min lr `1.00e-05`, `gae_lambda=1.0`, `ppo_clip_range=0.2`.
- `TRPO`: gamma `0.99`, lr `7.50e-05`, replay `120000`, batch `128`, target update `200`, replay warmup `6000`, cadence `32`, hidden `256,256`, activation `ReLU`, LR strategy `linear`, LR decay `0.4`, min lr `1.00e-05`, `gae_lambda=0.95`, `ppo_clip_range=0.2`.
- `SAC`: gamma `0.99`, lr `1.00e-04`, replay `200000`, batch `128`, target update `200`, replay warmup `10000`, cadence `32`, hidden `256,256`, activation `ReLU`, LR strategy `cosine`, LR decay `0.3`, min lr `1.00e-05`, `gae_lambda=0.95`, `ppo_clip_range=0.2`.

## Continuous algorithm status (implemented)
- `PPO`: on-policy trajectory updates with clipped surrogate objective, GAE, normalized advantages, and minibatch optimization.
- `A2C`: dedicated true actor-critic on-policy path (advantage-weighted policy gradient + value regression), separate from PPO/TRPO objective logic.
- `TRPO`: dedicated trust-region path with KL constraint, conjugate-gradient direction solve, and backtracking line search; critic is fitted separately.
- `SAC`: dedicated true off-policy path with continuous replay, twin critics + twin target critics, reparameterized actor, and learnable entropy temperature (`alpha`).

## GUI state (implemented)
- Environment panel renders live LunarLander frames.
- Parameter panel includes:
  - Environment controls (`continuous`, `gravity`, `wind`, `wind power`, `turbulence`, animation controls)
  - Compare controls with parameter dropdown + comma-separated values
  - General/specific hyperparameters with scientific notation fields where required.
- Controls row includes project-standard buttons plus runtime device button:
  - `Current device: CPU` by default
  - toggles to `Current device: GPU` when switched and CUDA is available
  - switching is blocked while training is active; existing agents are reset after switch.

## Compare mode behavior (implemented)
- Compare mode supports parameter-list based Cartesian combinations (not only policy comparison).
- Each combination launches as its own run key with immutable per-run metadata.
- Live plot and legend are updated per run key during compare execution and on finalization.
- Legend entries remain stable across pause/restart/new run flows and are not overwritten by mutable UI state.

## Training/runtime behavior
- `Train and Run` always starts with fresh agent networks for that launch.
- Worker thread runs training; GUI updates are coalesced on main thread.
- `Reset All` and `Clear Plot` are safe around running/paused workflows and avoid empty-legend warnings.
- Status line shows live optimizer LR in scientific notation.

## Validation state
- Logic and GUI regression tests include coverage for:
  - compare metadata integrity,
  - compare combination generation,
  - continuous cadence semantics,
  - PPO config propagation,
  - true `SAC`, `TRPO`, and `A2C` update-path smoke behavior.





