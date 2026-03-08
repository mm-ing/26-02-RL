# CarRacing Requirements Matrix

## Project-Specific Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Backend | SB3 backend required | ✅ Implemented | Uses `stable-baselines3` and `sb3-contrib` only. |
| Environment | `CarRacing-v3` with runtime params | ✅ Implemented | `lap_complete_percent`, `domain_randomize`, `continuous` exposed. |
| Render Mode | GUI environment render mode `rgb_array` | ✅ Implemented | GUI and rollout capture use `rgb_array`. |
| Policy List | `PPO`, `SAC`, `TD3`, `DDQN`, `QR-DQN` | ✅ Implemented | Exposed in logic + GUI. |
| Continuous Auto-Switch | Continuous mode tracks policy family | ✅ Implemented | Discrete for `DDQN`/`QR-DQN`; continuous otherwise. |
| Default Episodes | Default episodes = 3000 | ✅ Implemented | `DEFAULT_GENERAL['episodes'] = 3000`. |

## GUI Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Layout | Environment/Parameters top row with 2:1 ratio | ✅ Implemented | Top-level grid uses column weights `2:1`. |
| Parameters Group Order | Environment, Compare, General, Specific, Live Plot | ✅ Implemented | Order follows prompt. |
| Environment Controls | Runtime fields + Update + project env params | ✅ Implemented | Includes `lap_complete_percent`, `domain_randomize`, `continuous`. |
| Specific Group | Policy switch + dynamic policy-specific fields | ✅ Implemented | Per-policy cache/restore hardened with explicit active-policy tracking on switch. |
| Continuous Mode Auto-Switch | Continuous env mode from selected policy | ✅ Implemented | Discrete for `DDQN` and `QR-DQN`; continuous for others. |
| Controls Row | 8 controls in specified order | ✅ Implemented | Includes `Device` selector (`CPU`/`GPU`). |
| Current Run | Step/Episode progress and status line | ✅ Implemented | Steps progress is advanced during replay animation playback; `Episodes` row is positioned below `Steps`; status format matches contract. |
| Live Plot | Reward + MA + eval lines | ✅ Implemented | Legend outside plot; run history retained until clear. |
| Animation Replay | Multi-frame playback with latest-wins pending queue | ✅ Implemented | Active playback not interrupted; one pending slot. |
| Compare Mode | Add/Clear compare params and value combinations | ✅ Implemented | Includes Enter-to-add, comma-token Tab completion hints (`Policy`/`Activation`/`LR strategy` aliases), and compare-toggle animation-off behavior. |
| Legend Interactivity | Toggle + hover + scroll + visibility persistence | ✅ Implemented | Click text/handle toggles runs, hover cursor affordance, wheel scroll in legend area, hidden state persists across redraws. |
| Parameters Panel Scrollbar | Show vertical scrollbar only if needed | ✅ Implemented | Scrollbar is dynamically shown/hidden based on content height versus viewport height. |
| Parameter Field Sizing | Consistent width independent of label length | ✅ Implemented | Uses fixed label column + shared expandable input column for consistent field sizing. |
| Plot Theme Tone | Grid/spine/tick/label/legend text aligned to dark GUI tone | ✅ Implemented | Plot theme now applies dark-tone labels, ticks, spines, grid, and legend text consistently. |

## Logic Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---|---|
| Architecture | Env wrapper + policy wrapper/factory + trainer | ✅ Implemented | `CarRacingEnvWrapper`, `SB3PolicyFactory`, `CarRacingTrainer`. |
| Policy Exposure | `PPO`, `SAC`, `TD3`, `DDQN`, `QR-DQN` | ✅ Implemented | `QR-DQN` via `sb3-contrib`. |
| CNN Policies | Use CNN architecture | ✅ Implemented | All policies use `CnnPolicy`. |
| Shared NN Params | `hidden_layer`, LR strategy controls | ✅ Implemented | Wiring in model construction via `policy_kwargs` + schedule. |
| Train Loop | Background-safe loop with pause/cancel | ✅ Implemented | Uses pause/stop events and callback gating. |
| Event Contract | `step`, `episode`, `training_done`, `error` payloads | ✅ Implemented | Events include `run_id` + `session_id`; `step` events now emitted during train loop. |
| Eval Checkpoints | Deterministic eval every 10 episodes | ✅ Implemented | Appended into `eval_points`. |
| CSV Export | Transition export support | ✅ Implemented | Export to `results_csv/`. |
| Compare Parallelism | Bounded concurrent workers | ✅ Implemented | Worker cap is 4 and each run uses an isolated trainer instance. |
| Compare Defaults | If `Policy` compared, start from that policy defaults | ✅ Implemented | Compare config expansion now starts from selected policy defaults when `Policy` is compared. |
| Compare Key Applicability | Ignore policy-incompatible compare keys safely | ✅ Implemented | Overrides are applied only for keys valid in selected policy parameter schema. |
| CPU Budgeting | Adaptive per-worker CPU thread budgeting | ✅ Implemented | Per-run `cpu_thread_budget` is derived from CPU cores and active worker budget and applied in logic. |

## Test Coverage

| Area | Status | Notes |
|---|---|---|
| Logic utility functions | ✅ Implemented | Hidden-layer parse and policy mode tests. |
| Trainer behavior | ✅ Implemented | `run_episode` step count with dummy env. |
| GUI smoke | ✅ Implemented | GUI construction/destruction test with Tk guard. |
| GUI regressions | ✅ Implemented | Added tests for compare Tab-completion, compare toggle animation-off, legend visibility persistence, legend hover cursor affordance, legend scroll bound clamping, readonly combobox selectors for `activation`/`lr_strategy`, and specific-group separator placement. |
| Logic contract regressions (pause/cancel/eval cadence/compare applicability/policy isolation) | ✅ Implemented | Includes compare applicability, policy isolation, pause/resume/cancel finalization assertions, and deterministic eval-cadence checks. |

## High-Impact Open Gaps

- None identified in current contract audit scope; continue maintaining parity with future contract updates.

Latest isolated test outcome: `16 passed`.
