# Humanoid Requirements Matrix

## GUI Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Layout | Blueprint exact 4-row top-level layout and 2:1 top ratio | ✅ Implemented | Top-level grid uses row structure (`Environment/Parameters`, `Controls`, `Current Run`, `Live Plot`) with 2:1 top ratio. |
| Environment group | Runtime fields for Humanoid env below update button | ✅ Implemented | `Humanoid_gui.py` exposes all required Humanoid environment keys. |
| Specific group | Shared + policy-specific controls with policy snapshot isolation | ✅ Implemented | Shared controls and dynamic policy-specific frames with per-policy snapshots. |
| Controls row | Exact 8 controls and highlight-state behavior | ✅ Implemented | 8 controls in required order; train/pause highlight behavior and paused-restart behavior implemented. |
| Live plot | Reward/MA/eval plotting with legend outside right | ✅ Implemented | Axis labels, reward + MA + eval lines, legend outside right gutter. |
| Animation queue | Non-blocking playback and latest-wins pending slot | ✅ Implemented | Active playback uninterrupted; pending frame sequence replaced by latest. |
| Styling baseline | Full dark theme tokens and combobox/list styling contract | ✅ Implemented | Dark tokens applied for frames/inputs/buttons/progress and combobox popup listbox colors are explicitly configured; mousewheel cycling is disabled. |
| Compare group | Compare UI and compare execution behavior | ✅ Implemented | Compare on/Add/Clear, parameter/value cartesian runs, max 4 concurrent workers, Tab completion, and summary lines are implemented. |

## Logic Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Module split | Environment wrapper + policy factory + trainer | ✅ Implemented | `HumanoidEnvWrapper`, `SB3PolicyFactory`, `HumanoidTrainer`. |
| Trainer API | `run_episode`, `train`, `evaluate_policy`, env update | ✅ Implemented | Required methods implemented. |
| Event contract | `episode`, `training_done`, `error` with run/session IDs | ✅ Implemented | Payload includes IDs and key metrics. |
| Policy exposure | Exactly `PPO`, `SAC`, `TD3` | ✅ Implemented | Matches project file policy list. |
| LR scheduling | `constant`/`linear`/`exponential` with floor/decay | ✅ Implemented | Schedule wired in policy factory. |
| Hidden-layer wiring | Single-width and comma-separated parser | ✅ Implemented | Parsed architecture consumed in `policy_kwargs`. |
| Device fallback | GPU request falls back to CPU if CUDA unavailable | ✅ Implemented | Runtime resolver handles fallback safely. |
| Deterministic eval cadence | Fixed checkpoint cadence | ✅ Implemented | Controlled by `deterministic_eval_every` (default 10). |
| Compare execution | Cartesian combinations, max 4 workers | ✅ Implemented | GUI builds cartesian combinations with `expand_compare_runs`; workers are bounded via semaphore (`max 4`). |

## Test Coverage

| Area | Status | Notes |
|---|---|---|
| Logic smoke + event propagation | ✅ Implemented | Tests use fake env/model and verify event emissions. |
| Pause/resume/cancel | ✅ Implemented | Trainer event states are validated. |
| `run_episode` executed step count | ✅ Implemented | Confirms true executed steps and frame capture. |
| GUI policy snapshot regression | ✅ Implemented | Validates per-policy value isolation/restoration. |
| Isolated pytest setup | ✅ Implemented | Local `pytest.ini` with `testpaths = tests`. |

## High-Impact Open Gaps

- `🟡` Optional UX polish remains around long-legend readability when many compare dimensions are active (non-blocking, but can be refined further).

## Final Blueprint Recheck

Recheck sources:
1. `Johannes/Prompts/general.md`
2. `Johannes/Prompts/logic.md`
3. `Johannes/Prompts/gui.md`
4. `Johannes/Humanoid/Humanoid.md`

Result summary:
- Core architecture and SB3 backend requirements are in place.
- Humanoid environment and required policies are in place.
- Compare mode, strict visual update mode, and interactive legend behavior are in place.
- Compare legend enrichment dedup is in place; remaining gap is limited to optional UX polish for very large compare legends.
