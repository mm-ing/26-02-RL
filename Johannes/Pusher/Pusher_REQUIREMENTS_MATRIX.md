# Pusher Requirements Matrix

## GUI Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Top-level layout | Row0 `Environment` + `Parameters` (2:1), row1 `Controls`, row2 `Current Run`, row3 `Live Plot` | ✅ Implemented | Layout rewritten to 4-row contract and 2:1 top ratio. |
| Parameters panel | Scrollable panel with ordered groups (`Environment`, `Compare`, `General`, `Specific`, `Live Plot`) | ✅ Implemented | Scrollable canvas + ordered group builders. |
| Environment group | `Animation on`, `Animation FPS`, `Update rate`, `Frame stride`, `Update` + project env params in two columns | ✅ Implemented | Includes all required fields and Pusher params. |
| Environment rendering | Dedicated canvas, dark baseline, centered aspect-preserving redraw, main-thread rendering | ✅ Implemented | Canvas uses `#111111`; resize/center rendering path implemented. |
| Compare UI | Toggle + `Clear`/`Add`, parameter+values row, Enter-add, summary lines | ✅ Implemented | Compare controls and summary implemented. |
| Compare completion UX | Tab-completion + preview hint for categorical values | ✅ Implemented | Policy/activation/lr_strategy completion + hint + caret placement. |
| Compare runtime rules | Auto-disable animation on compare-on, one render run in compare, latest-wins pending replay | ✅ Implemented | Compare toggle disables animation; render policy selected; pending playback replacement used. |
| Specific group | Shared rows, separator, dynamic policy-specific rows, policy value isolation | ✅ Implemented | Shared/specific split with per-policy snapshots and restore. |
| Controls row | 8 controls in required order including device selector | ✅ Implemented | Buttons and selector match required order and count. |
| Current run panel | Steps/episodes progress bars + status format | ✅ Implemented | Both progress bars and status format implemented. |
| Live plot baseline | Axes labels, reward + MA + eval drawing, right legend gutter, persistent run history | ✅ Implemented | Plot contract baseline implemented with immutable run snapshot labels. |
| Live plot advanced legend UX | Interactive toggle, hover affordance, wheel scrolling on overflow, visibility persistence across redraws | ✅ Implemented | Pick-based toggle + hover cursor affordance + legend wheel-scroll offset + persisted visibility map across redraws. |
| Styling baseline | Dark tokens/theme and button/progress/input styling | 🟡 Partial | Core dark styling and button/progress themes applied; combobox popup/listbox styling parity may vary by platform. |
| Controls highlight behavior | Train/pause highlight transitions including paused restart behavior | ✅ Implemented | Train/pause styles switch by run state; paused train triggers fresh run path. |

## Logic Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Backend | SB3-only algorithms | ✅ Implemented | Uses SB3 `PPO`, `SAC`, `TD3`, `DDPG`. |
| Architecture | Environment wrapper + policy wrapper + trainer | ✅ Implemented | `PusherEnvironment`, `SB3PolicyAgent`, `PusherTrainer`. |
| Trainer API | `run_episode`, `train`, `evaluate_policy`, env update/rebuild support | ✅ Implemented | Required APIs present and used. |
| Event contract | `step`, `episode`, `training_done`, `error` with required payload fields | ✅ Implemented | Step and episode events emitted; episode includes required metrics and frames. |
| Eval cadence | Deterministic eval every 10th episode and final | ✅ Implemented | Cadence enforced in train loop. |
| NN controls | Hidden-layer parsing + LR strategy wiring | ✅ Implemented | Net arch and LR schedules consumed by model construction. |
| Device fallback | GPU selection with CUDA fallback to CPU | ✅ Implemented | Logic resolves effective device safely. |
| Compare mode | Cartesian combinations, max 4 workers, policy-baseline overrides, per-worker CPU threads | ✅ Implemented | Compare config builder + GUI bounded worker executor + CPU thread budgeting. |
| Exports | CSV and PNG exports with run-parameterized naming | ✅ Implemented | CSV via trainer; PNG via GUI training completion and manual button. |

## Test Coverage

| Area | Status | Notes |
|---|---|---|
| Logic helpers | ✅ Implemented | Hidden-layer parser and LR floor tests. |
| Training event cadence | ✅ Implemented | Verifies `step` + `episode` emissions and eval cadence. |
| Pause/resume/cancel | ✅ Implemented | Dedicated transition test validates final canceled completion event. |
| Transition CSV export | ✅ Implemented | Verifies non-empty CSV with expected columns. |
| Compare baseline/override compatibility | ✅ Implemented | Policy-default baseline and incompatible-key ignore tested. |
| GUI startup/layout smoke | ✅ Implemented | Confirms required panels and controls exist. |
| GUI policy isolation | ✅ Implemented | Policy switches preserve independent shared/specific values. |
| GUI compare input UX | ✅ Implemented | Compare add/clear, tab completion, hint, caret placement covered. |
| GUI stale-event and queue behavior | ✅ Implemented | Stale session ignore and animation latest-wins queue behavior tested. |
| GUI advanced legend interaction regressions | 🟡 Partial | Toggle persistence and hover affordance tests added; overflow scroll interaction covered in runtime logic but lacks direct automated assertion. |

## High-Impact Open Gaps

1. Platform-dependent combobox popup/listbox dark styling parity may not fully match the strict style token contract on all Tk builds.
2. Legend overflow wheel-scrolling behavior is implemented but still missing a direct deterministic GUI automation assertion.

## Mandatory Final Contract Recheck

- `general.md`: output structure, stack, env guards, threading guardrails, and requirements baseline are implemented.
- `logic.md`: trainer architecture, event contract, compare execution behavior, device handling, exports, and required logic tests are implemented.
- `gui.md`: major layout/controls/groups/progress/compare/runtime and legend interaction contract are implemented; residual risk is mainly platform-level style parity and a remaining overflow-scroll test gap.
- `Pusher.md`: environment id/params, SB3 backend, policy exposure, and specific policy panels are implemented.

Final status: `🟡 Partial` due to remaining advanced legend interaction/styling parity gaps.
