# LunarLander Project Specification (Current State)

## Goal
Maintain and evolve the `LunarLander` RL project based on Gymnasium `LunarLander-v3`, following the architecture and GUI baseline from `RL_initial_2.0.md`, with project-specific behavior documented here.

## Implementation requirement (authoritative)
- All policies in this project must use a `stable_baselines3`/`sb3-contrib` backend (PyTorch).
- Do not implement policy training with TensorFlow/Keras backends in this project.
- If `TRPO` from `sb3-contrib` is unavailable, use the documented fallback (`PPO`) and document that fallback in logs/docs.

## Backend architecture
- Training backend is based on `stable_baselines3` (PyTorch backend, not TensorFlow).
- Policy mapping:
  - `DuelingDQN`, `D3QN`, `DDQN+PER` -> SB3 `DQN`
  - `PPO` -> SB3 `PPO`
  - `A2C` -> SB3 `A2C`
  - `TRPO` -> `sb3-contrib` `TRPO` when available (fallback to SB3 `PPO`)
  - `SAC` -> SB3 `SAC`
- Existing GUI-facing `Trainer` API is preserved (`run_episode`, `train`, `evaluate_policy`, `set_policy_config`, etc.).
- Legacy custom fallback policy-training paths were removed; training is now exclusively SB3/SB3-contrib based.

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
  - GUI field `continous` and environment mode are synchronized when automatic mode switching is required.

## Current default presets
- `DuelingDQN`: gamma `0.99`, lr `3.00e-04`, replay `100000`, batch `128`, target update `200`, replay warmup `5000`, cadence `2`, hidden `256,256,128`, activation `ReLU`, LR strategy `exponential`, LR decay `0.1`, min lr `1.00e-05`.
- `D3QN`: gamma `0.99`, lr `2.50e-04`, replay `150000`, batch `128`, target update `200`, replay warmup `8000`, cadence `2`, hidden `512,256,128`, activation `ReLU`, LR strategy `exponential`, LR decay `0.1`, min lr `1.00e-05`.
- `DDQN+PER`: gamma `0.99`, lr `2.00e-04`, replay `200000`, batch `128`, target update `200`, replay warmup `10000`, cadence `2`, hidden `512,256,128`, activation `ReLU`, LR strategy `exponential`, LR decay `0.1`, min lr `1.00e-05`.
- `PPO`: gamma `0.99`, lr `3.00e-04`, replay `100000`, batch `128`, target update `200`, replay warmup `5000`, cadence `256`, hidden `256,256`, activation `ReLU`, LR strategy `linear`, LR decay `0.3`, min lr `1.00e-05`, `gae_lambda=0.95`, `ppo_clip_range=0.2`.
- `A2C`: gamma `0.99`, lr `3.00e-04`, replay `100000`, batch `128`, target update `200`, replay warmup `5000`, cadence `64`, hidden `256,256`, activation `ReLU`, LR strategy `exponential`, LR decay `0.3`, min lr `1.00e-05`, `gae_lambda=1.0`, `ppo_clip_range=0.2`.
- `TRPO`: gamma `0.99`, lr `1.00e-04`, replay `120000`, batch `64`, target update `200`, replay warmup `6000`, cadence `256`, hidden `256,256`, activation `ReLU`, LR strategy `linear`, LR decay `0.4`, min lr `1.00e-05`, `gae_lambda=0.95`, `ppo_clip_range=0.2`.
- `SAC`: gamma `0.99`, lr `1.00e-04`, replay `200000`, batch `128`, target update `200`, replay warmup `10000`, cadence `32`, hidden `256,256`, activation `ReLU`, LR strategy `cosine`, LR decay `0.3`, min lr `1.00e-05`, `gae_lambda=0.95`, `ppo_clip_range=0.2`.

## Continuous algorithm status (implemented)
- `PPO`, `A2C`, `TRPO`, and `SAC` are provided through the SB3/SB3-contrib backend.
- GUI parameters are translated into SB3 model constructor parameters where applicable.

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
- If `Policy` is one compared parameter, each run automatically applies that policy's defaults for non-compared policy hyperparameters.
- Live plot and legend are updated per run key during compare execution and on finalization.
- Legend entries remain stable across pause/restart/new run flows and are not overwritten by mutable UI state.

## Deterministic evaluation tracking (implemented)
- During training, periodic deterministic evaluations are executed (no exploration sampling, no learning updates).
- Evaluation uses deterministic action selection (`argmax` for discrete policies, actor mean/tanh path for continuous policies).
- Eval checkpoints are tracked separately from training rewards.
- Finalized plot runs include an additional `eval` line in the legend (`<base label> | eval`) alongside `reward` and `MA`.

## Training/runtime behavior
- `Train and Run` always starts with fresh agent networks for that launch.
- Worker thread runs training; GUI updates are coalesced on main thread.
- `Reset All` and `Clear Plot` are safe around running/paused workflows and avoid empty-legend warnings.
- Status line shows live optimizer LR in scientific notation.
- Deterministic evaluation episodes are side-effect free for training state (no replay insertions, no optimizer steps, no end-of-episode learning flush).

## Validation state
- Logic and GUI regression tests include coverage for:
  - compare metadata integrity,
  - compare combination generation,
  - compare-policy default propagation for non-compared fields,
  - deterministic eval side-effect safety,
  - continuous cadence semantics,
  - PPO config propagation,
  - true `SAC`, `TRPO`, and `A2C` update-path smoke behavior.





