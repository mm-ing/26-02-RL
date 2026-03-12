# Ant Requirements Matrix

## GUI Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Layout | Top-level row/column layout and panel order | ✅ Implemented | Environment/Parameters, Controls, Current Run, Live Plot are present in requested order. |
| Environment group | Animation controls + env parameter controls + update button | ✅ Implemented | Uses Reacher-style pair grid (`label,input | label,input`) with Ant-specific env weights below Update. |
| Compare group | Compare on, add/clear, parameter/value input, summaries | ✅ Implemented | Tab completion is implemented for categorical fields. |
| Specific group | Policy selector, shared params, dynamic policy-specific params | ✅ Implemented | SAC/TQC/CMA-ES controls are policy-aware, cached per policy, and rendered in Reacher-style pair grid (`label,input | label,input`). |
| Controls row | 8 controls in requested order including device selector | ✅ Implemented | Includes pause/run behavior and reset/clear/save actions. |
| Event bridge | Queue-based worker event pump with session tags | ✅ Implemented | Queue is flushed before new sessions and stale events are filtered by session ID. |
| Animation queue | Latest-wins pending playback while active playback runs | ✅ Implemented | One pending buffer slot is maintained. |
| Live plot details | Interactive legend persistence and advanced overflow scrolling | ✅ Implemented | Click-to-toggle is implemented for legend text/handles, with hover affordance (cursor + entry visual change) and wheel scrolling when legend overflows. |

## Logic Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Architecture | Environment wrapper, policy wrapper, trainer class | ✅ Implemented | `AntEnvironment`, `AntPolicyAgent`, `AntTrainer` in logic module. |
| Policies | Expose SAC, TQC, CMA-ES | ✅ Implemented | SAC/TQC via SB3; CMA-ES via EvoTorch with fallback path. |
| Events | `episode`, `training_done`, `error` and required payload fields | ✅ Implemented | Session/run tags and baseline payload fields emitted. |
| LR/NN controls | hidden_layer parsing + LR strategy scheduling | ✅ Implemented | `parse_hidden_layer` and `build_lr_schedule` wired into SB3 model kwargs. |
| Runtime | Pause/resume/cancel for active trainers | ✅ Implemented | Implemented and bridged from GUI to all active workers; paused Train-and-Run now starts a fresh run. |
| Compare mode | Cartesian runs, bounded parallelism <=4 | ✅ Implemented | Uses `expand_compare_runs`, max 4 workers, adaptive per-worker CPU thread budget, and single selected render-run animation. |
| Exports | CSV transitions and PNG plots | ✅ Implemented | CSV from trainer data and PNG from live plot panel. |
| Deterministic eval cadence | Evaluate every 10 episodes when enabled | ✅ Implemented | `evaluation_rollout_on` controls periodic eval checkpoints. |

## Test Coverage

| Area | Status | Notes |
|---|---|---|
| Logic core utilities | ✅ Implemented | Hidden-layer parsing, LR schedule floor, compare expansion tested. |
| Trainer event flow | ✅ Implemented | Episode, episode_aux, and training_done flows are tested via monkeypatched run loop. |
| GUI smoke startup/reset | ✅ Implemented | Tk startup and reset/clear smoke path included. |
| Policy snapshot isolation | ✅ Implemented | Basic policy switch value persistence test included. |
| Advanced GUI regressions | ✅ Implemented | Includes stale-session filtering, compare render selection, paused restart, combobox wheel guard, split-event flow, step-event handling, legend hover affordance, and resume-before-cancel shutdown behavior tests. |

## High-Impact Open Gaps

- EvoTorch CMA-ES path includes compatibility fallback and dedicated searcher-path test coverage; a true full-stack Ant-v5 + EvoTorch integration run should still be validated in a fully provisioned MuJoCo runtime.
