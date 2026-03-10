# Pusher Requirements Matrix

## GUI Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Environment group | Runtime inputs for `reward_near_weight`, `reward_dist_weight`, `reward_control_weight` below update button in two columns | ✅ Implemented | See `Pusher_gui.py` environment section. |
| Specific group | Per-policy specific input panels and shared NN/LR controls | ✅ Implemented | Includes shared + policy-specific vars and snapshot restore by policy. |
| Runtime responsiveness | Background thread training and UI event pump; no worker-side drawing | ✅ Implemented | Queue-based worker events with `after()` UI pump. |
| Animation behavior | Non-blocking playback and latest-wins pending buffer | ✅ Implemented | Active playback is uninterrupted; pending sequence overwritten by newest. |
| Control surface | Train, pause, resume, cancel controls | ✅ Implemented | Buttons call trainer control events. |

## Logic Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Module split | Environment wrapper + policy factory + trainer | ✅ Implemented | `PusherEnvWrapper`, `SB3PolicyFactory`, `PusherTrainer`. |
| Trainer API | `run_episode`, `train`, `evaluate_policy`, env update | ✅ Implemented | All required trainer methods are present. |
| Event contract | `episode`, `training_done`, `error` with `session_id`/`run_id` | ✅ Implemented | Event payload includes required identifiers and reward fields. |
| Episode payload content | Includes reward stats, eval points, steps, lr, best reward, render state, frames | ✅ Implemented | Emitted each episode with frame gating by update rate. |
| Deterministic eval cadence | Fixed interval (default every 10th episode) | ✅ Implemented | Controlled by `deterministic_eval_every` config. |
| Policy exposure | `PPO`, `SAC`, `TD3`, `DDPG` | ✅ Implemented | Exposed in logic and GUI. |
| Device handling | CPU/GPU option with safe fallback | ✅ Implemented | GPU requires CUDA availability else CPU. |
| LR scheduling | Constant/linear/exponential with `min_lr`/`lr_decay` | ✅ Implemented | Schedule consumed in SB3 model creation. |
| Hidden-layer parsing | Single value -> symmetric two layers, CSV -> direct list, fallback on invalid | ✅ Implemented | Implemented in policy factory parser. |
| CSV export | Optional transition export to `results_csv/` | ✅ Implemented | Triggered when `export_csv` is enabled. |

## Test Coverage

| Area | Status | Notes |
|---|---|---|
| Logic smoke + event propagation | ✅ Implemented | Tests use fake env/model to validate event emissions and payload shape. |
| Pause/resume/cancel | ✅ Implemented | Unit test validates state transitions. |
| `run_episode` step count behavior | ✅ Implemented | Verifies actual step counts and done handling. |
| GUI policy snapshot behavior | ✅ Implemented | Checks policy switch snapshot isolation behavior. |
| Isolated pytest setup | ✅ Implemented | Local `pytest.ini` with `testpaths = tests`. |

## High-Impact Open Gaps

- `🟡` Compare mode Cartesian multi-worker execution (up to 4 workers) is not implemented in this initial setup.
- `🟡` CNN policy architecture is not used because `Pusher-v5` provides vector observations; current implementation uses MLP policy wiring.
