# Reacher Requirements Matrix

## GUI Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Layout | Environment/Parameters top row 2:1, controls/current/live rows | ✅ Implemented | Grid layout and 8-control row order match contract. |
| Layout | Parameter inputs use two-column arrangement | ✅ Implemented | General/Environment/Specific sections use paired label-input columns; covered by GUI regression test. |
| Parameters | Group ordering and core fields | ✅ Implemented | Environment, Compare, General, Specific, Live Plot order is correct. |
| Parameters | Compare dropdown includes `General + Specific` (exclude Environment) | ✅ Implemented | Compare dropdown now includes `max_steps` and `episodes` plus shared/specific keys. |
| Parameters | Tooltip hints for training parameters | ✅ Implemented | Short hover tooltips added for environment/general/specific training controls. |
| Current Run | Steps progress advances during replay animation only | ✅ Implemented | Step row is updated in playback path and ignored on worker `step` events. |
| Thread Bridge | Queue + `after()` pump, stale session filtering | ✅ Implemented | `session_id` tagging/filtering implemented in GUI bridge and pump. |
| Animation | Non-blocking playback + latest-wins pending slot | ✅ Implemented | Active playback is uninterrupted and pending slot is overwritten by newest payload. |
| Compare Render | Exactly one compare worker feeds animation/render | ✅ Implemented | Compare render worker is selected once (`selected policy` preferred, else first). |
| Plot | Reward/MA/eval styling + right legend gutter + persistent run visibility | ✅ Implemented | Core line styles, gutter margins, and visibility state persistence are in place. |
| Plot | Legend handle/text click toggles + overflow wheel scrolling | ✅ Implemented | Legend toggles now map text+handle for run labels and support bounded wheel scrolling while hovered. |
| Styling | Dark baseline colors and neutral button styling | ✅ Implemented | Palette, button states, progressbar, and dark inputs are set. |

## Logic Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Architecture | Env wrapper + SB3 agent + trainer separation | ✅ Implemented | `ReacherEnvironment`, `SB3PolicyAgent`, `ReacherTrainer` exist and are reusable headless. |
| Event Contract | `step`, `episode`, `episode_aux`, `training_done`, `error` | ✅ Implemented | Events and key payload fields are emitted; GUI bridge adds `session_id`. |
| SB3 Policies | Expose `PPO`, `SAC`, `TD3` | ✅ Implemented | Mapping and defaults include required policies. |
| Shared NN Params | hidden layers, activation, LR strategy/min/decay wiring | ✅ Implemented | Parsed and wired into SB3 config and LR schedule. |
| Device | CPU/GPU selection with safe fallback | ✅ Implemented | GPU request falls back to CPU when CUDA unavailable. |
| Runtime | Flush stale queued worker events before new run | ✅ Implemented | `_flush_event_queue()` is invoked before launching fresh single/compare runs. |
| Compare Runtime | Cartesian compare runs, max workers 4 | ✅ Implemented | Combo expansion with `ThreadPoolExecutor(max_workers=4)`. |
| Compare Runtime | Policy compare starts from selected policy defaults only | ✅ Implemented | Policy-compare mode can build from policy defaults baseline, then applies explicit compared values. |
| Compare Runtime | Apply only policy-compatible compare keys | ✅ Implemented | Compare keys are filtered by policy-specific allow-list before assignment. |
| Compare Runtime | Exactly one selected render run in compare mode | ✅ Implemented | Single render index is selected and enforced per compare session. |
| Compare Runtime | Adaptive per-worker CPU thread budgeting | ✅ Implemented | Worker thread count budget is derived from CPU cores and compare worker count. |
| Exports | CSV transitions + PNG plots | ✅ Implemented | CSV and PNG export paths are implemented. |

## Test Coverage

| Area | Status | Notes |
|---|---|---|
| Logic smoke/tests | ✅ Implemented | Parser, schedule, run_episode steps, eval cadence, CSV export, pause/cancel checks exist. |
| GUI smoke/tests | ✅ Implemented | Startup smoke, policy cache isolation, compare tab completion, compare options, render-index, playback-progress, policy-default baseline, stale-session filter, split-event behavior, and two-column parameter layout checks exist. |
| Contract regressions from `gui.md`/`logic.md` | ✅ Implemented | Includes stale-event regression and split `episode`/`episode_aux` regression in GUI tests. |
| Isolated pytest run | ✅ Implemented | Isolated run passes: `16 passed`. |

## High-Impact Open Gaps
- None currently identified from the contract files after final recheck.
