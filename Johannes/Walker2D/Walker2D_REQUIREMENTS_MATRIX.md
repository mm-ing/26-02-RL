# Walker2D Requirements Matrix (Strict Pass)

Scope audited:
- `Walker2D_gui.py`
- `Walker2D_logic.py`
- `tests/test_Walker2D_gui.py`
- `tests/test_Walker2D_logic.py`

Status legend:
- ✅ Implemented
- 🟡 Partial
- ❌ Missing

## Future Project Rule

For every new RL project in this workspace:
1. Copy `REQUIREMENTS_MATRIX_TEMPLATE.md` to `<ProjectName>_REQUIREMENTS_MATRIX.md`.
2. Fill/update the matrix after each implementation pass.
3. Before final handoff, ensure the `High-Impact Open Gaps` section has no unresolved blockers.
4. Run isolated tests and record the pass result.

## GUI Contract Coverage (from `gui.md` + `Walker2D.md`)

| Area | Requirement | Status | Notes |
|---|---|---:|---|
| Architecture | Main GUI with env panel, params panel, controls, current run, live plot | ✅ | Implemented with required top-level groups and 2:1 top ratio. |
| Thread bridge | Queue + `after()` pump; event types incl. `step`, `episode`, `training_done`, `error` | ✅ | Implemented; stale session filtering in `_poll_worker_events`. |
| Worker registry | Thread-safe active worker registry + pre-registration | ✅ | Lock-protected `active_trainers`; pre-registered in compare path. |
| Environment panel | Canvas dark background, centered/aspect-preserving redraw | ✅ | Implemented in `_draw_frame` and resize callback. |
| Animation replay | Full sequence playback at FPS, latest-wins pending queue | ✅ | One active + one pending playback slot behavior implemented. |
| Parameters panel | Scrollable, wheel only inside, scrollbar shown only when needed | ✅ | Implemented (`_bind_mousewheel`, `_on_params_configure`). |
| Parameters panel width | Parameter groups fill full parameter panel width | ✅ | Implemented with full-width inner frame/group column weighting and canvas-window width sync. |
| Parameter group order | Environment, Compare, General, Specific, Live Plot | ✅ | Implemented in `_build_parameters_panel`. |
| Environment fields | Animation on/FPS/update rate/frame stride + Update + env parameters | ✅ | Includes all Walker2D env parameters below Update. |
| Compare layout | Top row Compare/Clear/Add; second row Parameter + Values no extra labels | ✅ | Implemented. |
| Compare add/clear/enter | Add on button/Enter, clear list, summary lines `Parameter: [..]` | ✅ | Implemented. |
| Compare auto animation-off | Compare on disables animation by default | ✅ | Implemented in `_on_compare_toggle`. |
| Compare dropdown coverage | Include all General + Specific params, exclude Environment | ✅ | Dynamic options now include `episodes`, `max_steps`, shared params, and active policy-specific params. |
| Compare tab completion | Categorical suggestions with Tab + caret at end | ✅ | Implemented with typed-prefix completion, caret-at-end behavior, and dynamic preview hint updates. |
| Compare hint alignment | Hint aligned under values column | ✅ | Implemented. |
| Compare hint visibility | Completion preview only shown when typed prefix has matching categorical suggestion | ✅ | Implemented via conditional hint rendering in `_update_compare_hint`; covered by GUI regression test. |
| Compare run behavior | Cartesian combinations, max 4 workers | ✅ | Implemented with `itertools.product` + `ThreadPoolExecutor(max_workers<=4)`. |
| Compare render selection | Exactly one render run; selected policy preferred | ✅ | Implemented with `render_run_id` selection by policy. |
| General fields | Max steps + Episodes | ✅ | Implemented. |
| Specific shared/specific rows | Shared top, separator, dynamic policy-specific rows | ✅ | Implemented. |
| Policy switch value preservation | Changing policy must not overwrite current values | ✅ | Cached values preserved by previous-policy snapshot logic. |
| Specific-group policy isolation | All `Specific` parameters (shared + policy-specific) are independently adjustable per policy and restored on policy switch | ✅ | Implemented via per-policy caches for shared rows (`gamma`, `learning_rate`, `batch_size`, `hidden_layer`, `lr_strategy`, `min_lr`, `lr_decay`) and policy-specific rows; covered by dedicated GUI regressions. |
| NN shared fields | Hidden-layer and LR schedule controls are exposed and applied for all exposed NN policies | ✅ | Implemented shared controls: `hidden_layer` (single width or comma-separated architecture), `lr_strategy`, `min_lr`, `lr_decay` with compare and snapshot wiring. |
| Tooltips | Short hover tooltips for training-relevant params | ✅ | Hover tooltips added across environment/general/specific/live-plot/device controls. |
| Live plot group | Moving average + Show Advanced + advanced fields | ✅ | Implemented. |
| Controls row | Exactly 8 equal-width controls in required order | ✅ | Implemented. |
| Control highlight semantics | Train highlight active, pause highlight while paused, restart while paused starts fresh run | ✅ | Implemented. |
| Control neutral baseline | Inactive action buttons use explicit neutral style (no platform-default fallback) | ✅ | Implemented via `Neutral.TButton` defaults and style-state switching in `_set_control_styles`. |
| Device selector | CPU/GPU selector default CPU | ✅ | Implemented. |
| Current run status | Steps/Episodes bars + status format | ✅ | Implemented. |
| Steps progress semantics | Steps progress updates during replay animation | ✅ | Implemented via playback path. |
| Plot basics | No title, axis labels, reward alpha, MA/eval style & width | ✅ | Implemented. |
| Plot grid runtime | Live plot grid enabled in runtime configuration | ✅ | Implemented with explicit `ax.grid(True, ...)`; covered by GUI regression test. |
| Legend placement/gutter | Outside-right legend + reserved right gutter | ✅ | Implemented (`subplots_adjust right=0.78`). |
| Legend labels | `moving average`, `evaluation rollout` | ✅ | Implemented. |
| Legend interaction | Click toggle + hover affordance + hand cursor | ✅ | Implemented and tested. |
| Legend overflow scroll | Mouse wheel scrolling while hovering legend, bounded | ✅ | Implemented and tested. |
| Immutable run snapshots | Historical labels not overwritten by live controls | ✅ | Implemented via `run_meta_snapshots`. |
| Compare legend enrichment | Append compare params not already represented; avoid duplicates | ✅ | Implemented with field-aware duplicate filtering so base legend fields are not repeated in compare suffix. |
| Combobox wheel safety | Mouse wheel must not cycle dropdown selections | ✅ | Implemented for policy/device/compare comboboxes. |
| Animation-off runtime behavior | Disable animation clears queue/progress and render state | ✅ | Implemented in `_on_animation_toggle`. |

## Logic Contract Coverage (from `logic.md` + `Walker2D.md`)

| Area | Requirement | Status | Notes |
|---|---|---:|---|
| Backend | SB3-only with PPO/SAC/TD3 | ✅ | Implemented in model factory. |
| Env ID | `Walker2d-v5` gymnasium | ✅ | Implemented in `Walker2DEnvConfig`. |
| Runtime env params | Full Walker2D runtime-configurable env parameters | ✅ | Implemented and wired. |
| Trainer API | `run_episode`, `train`, `evaluate_policy`, env rebuild/update | ✅ | Implemented. |
| Event types | `episode`, `training_done`, `error` (+ `step` bridge) | ✅ | Implemented. |
| Episode payload | includes required keys + frames | ✅ | Implemented with `frames` and compatibility `frame`. |
| Deterministic eval cadence | Every 10th episode | ✅ | `deterministic_eval_every=10` used by default; deterministic evaluation steps are always coupled to `max_steps`. |
| Device behavior | CPU/GPU selectable with safe fallback | ✅ | GPU falls back to CPU if CUDA unavailable. |
| PPO constraints | `n_steps >= batch_size` + divisibility adjustment | ✅ | Implemented. |
| NN policy kwargs | Shared hidden-layer architecture applied to all SB3 policies via `policy_kwargs.net_arch` | ✅ | Implemented for PPO/SAC/TD3 with parsing rules: single value `256` -> `[256,256]`, comma-separated `256,128,64` -> `[256,128,64]` (with safe fallback to defaults). |
| LR schedule behavior | `lr_strategy`, `min_lr`, `lr_decay` control SB3 learning-rate schedule | ✅ | Implemented `constant`/`linear`/`exponential` schedule mapping in logic and applied to all exposed policies. |
| SAC/TD3 defaults | Realistic warmup/buffer defaults | ✅ | Implemented. |
| Pause semantics | `pause_event.set()` running, `clear()` paused, wait gates loops | ✅ | Implemented. |
| Latest-frame-wins | One active + one pending playback queue | ✅ | Implemented GUI-side. |
| Compare mode | Cartesian runs + max 4 concurrency | ✅ | Implemented. |
| Compare GPU rule | Do not force compare to CPU | ✅ | Implemented; uses selected device fallback rules. |
| Unique run IDs | Unique internal run IDs for compare combos | ✅ | Implemented. |
| Register worker handles early | Before compare execution | ✅ | Implemented. |
| CPU thread budgeting | Adaptive per-worker CPU thread budgeting by core count | ✅ | Implemented in compare workers with per-worker budgets derived from available CPU cores. |
| Compare policy defaults rule | If `Policy` compared, start from policy defaults then apply explicit overrides | ✅ | Implemented in compare config generation with policy-specific override applicability; regression test added. |
| CSV export | Export sampled transitions to `results_csv/` | ✅ | Implemented and GUI wired. |
| PNG export | Save plot to `plots/` with timestamp and params | ✅ | Filename includes policy, max steps, gamma, learning rate, and timestamp. |

## Test Coverage vs Prompted Robustness Tests

| Area | Status | Notes |
|---|---:|---|
| Logic smoke tests | ✅ | Basic run_episode + event completion tests present. |
| GUI smoke tests | ✅ | Startup/reset test present. |
| Legend interaction regression | ✅ | Click + overflow scroll test present. |
| Pause/resume/cancel regression | ✅ | Dedicated GUI regression verifies paused-run restart starts a fresh run and cancels old trainer. |
| Policy switch regression | ✅ | Dedicated GUI regression verifies compare dropdown updates with active policy-specific fields. |
| Specific parameter isolation regression | ✅ | Dedicated GUI regressions verify per-policy isolation for policy-specific keys (e.g. `tau`) and shared specific keys (`gamma`, `learning_rate`, `batch_size`, `hidden_layer`, `lr_strategy`, `min_lr`, `lr_decay`). |
| NN shared-field logic regression | ✅ | Dedicated logic regression verifies hidden-layer net-arch wiring and learning-rate schedule behavior. |
| Compare render/run finalization regressions | ✅ | Dedicated GUI regression verifies render-run selection preference and compare completion finalization sequencing. |
| Stale event regression | ✅ | Dedicated GUI regression verifies non-current session events are ignored in event pump. |
| Shutdown paused-worker regression | ✅ | Dedicated GUI regression verifies close path resumes paused workers before cancel and shuts down cleanly. |
| Compare thread budgeting regression | ✅ | Dedicated GUI regression verifies fair bounded per-worker budget distribution. |

## Highest-Impact Remaining Gaps

1. No high-impact open gaps remain in the current matrix scope.
