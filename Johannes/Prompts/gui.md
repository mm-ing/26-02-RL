# GUI Prompt Blueprint

## GUI Architecture
Implement one main GUI class with:
- environment rendering panel
- parameters panel (scrollable)
- controls row
- current-run status/progress
- live matplotlib plot
- background training threads with main-thread UI pump

Use this threading bridge:
- queue/event bridge between workers and GUI (`after()` polling)
- publish structured events (`step`, `episode`, `episode_aux`, `training_done`, `error`)
- tag worker-originated events with `session_id` in the GUI-side worker bridge and ignore stale session events in the pump
- pause/resume/cancel actions must control all active worker trainers for the current session (single-run and compare)
- keep the active worker registry thread-safe (lock reads/writes) and pre-register workers before launching jobs so pause works immediately
- emit `episode` as a fast metric event first; emit heavy payloads (`frames`, refreshed `eval_points`) via `episode_aux` so live plot progress is not blocked by replay/eval work

---

## GUI Layout (Must Match)
Top-level grid:
- row 0: `Environment` (left), `Parameters` (right)
- row 1: `Controls` (full width)
- row 2: `Current Run` (full width)
- row 3: `Live Plot` (full width)

Top row ratio:
- `Environment : Parameters = 2 : 1`

### Environment Panel
- dedicated render canvas
- canvas background baseline: `#111111`
- keep aspect ratio and center image
- redraw on resize
- render on main thread only
- for training animation, replay per-episode frame buffers captured during training at `Animation FPS` (no single-frame-only behavior)
- replay queue semantics: if a playback is active, keep it running and store only one pending playback slot; incoming newer playback replaces pending slot (`latest-wins`)
- playback must not block worker training loops; first episode playback can run while the next episodes continue training

### Parameters Panel
- scrollable content region
- make parameter groups fill full panel width
- group order:
  1. `Environment`
  2. `Compare`
  3. `General`
  4. `Specific`
  5. `Live Plot`
- mouse wheel active only while cursor is inside panel
- show vertical scrollbar only if needed

#### Environment Group Fields
- `Animation on` (default `True`)
- `Animation FPS` (default `30`)
- `Update rate (episodes)` (default `1`)
- `Frame stride` (default `2`; capture every Nth frame during replay rollout sampling)
- `Update` button
- environment-specific parameters below `Update` must come from the project-specific file (do not hard-code unrelated environment keys)

#### Compare Group Fields
- `Compare on` toggle
- top control row: `Compare on` at left, `Clear` and `Add` buttons at right (side-by-side)
- compare parameter dropdown (`General` + `Specific` parameters only; exclude `Environment`)
- compare values text input (comma-separated)
- second row layout: `Parameter` dropdown (left) and `Values` input (right), without additional field labels
- compare values suggestions + `Tab` completion for categorical params (`Policy`, `Activation`, `LR strategy`)
- compare values completion preview hint shown below the `Values` input (for example `Tab -> Tanh`)
- show completion preview hint only when the current `Values` input has a typed prefix and a matching categorical suggestion exists
- after `Tab` autofill, place the text cursor at the end of the inserted value in the `Values` input
- align the completion preview hint with the `Values` input column (not under the parameter dropdown column)
- summary lines: `Parameter: [v1, v2, ...]`

Rules:
- when compare is on, set animation off automatically (user may re-enable)
- `Add` commits selected parameter + values into active compare lists
- pressing `Enter` in compare values input triggers `Add`
- `Clear` removes all active compare parameter lists
- include all `General` + `Specific` parameters in compare dropdown; exclude `Environment`
- allow accepting categorical suggestion with `Tab` based on typed prefix
- when compare mode runs multiple workers, only one selected run should feed live animation/playback (selected policy preferred; fallback first run)
- if training outpaces playback, pending animation should keep only the newest completed episode buffer (overwrite older pending episode buffers)

#### General Group Fields
- `Max steps`
- `Episodes`

#### Specific Group Fields
- top row: `Policy`
- shared rows first (same parameter names across all exposed policies), preserving the existing two-column arrangement where possible
- for NN-based policies, shared rows must include at least: `gamma`, `learning_rate`, `batch_size`, `hidden_layer`, `activation`, `lr_strategy`, `min_lr`, `lr_decay`
- `hidden_layer` input accepts either a single width (for example `256`) or a comma-separated architecture (for example `256,128,64`)
- `activation` must be a selector (`ReLU`, `Tanh`)
- `lr_strategy` must be a selector (`constant`, `linear`, `exponential`)
- draw a horizontal separator below shared rows
- policy-specific rows below separator, updated dynamically when `Policy` changes
- include only parameters that are actually consumed by the selected policy in the backend configuration
- do not show irrelevant controls for the selected policy (for example exploration controls for deterministic continuous-control policies)
- keep row ordering stable for shared rows and stable per-policy for specific rows

Notes:
- use scientific notation for LR text inputs
- changing `Policy` should only change visible policy-specific rows/options; it must not overwrite current parameter values
- all parameters shown in `Specific` (shared rows + policy-specific rows) must be cached per policy and restored when switching back to that policy
- shared specific rows (`gamma`, `learning_rate`, `batch_size`, `hidden_layer`, `lr_strategy`, `min_lr`, `lr_decay`) must use per-policy values/defaults and must not leak across policies
- parameter defaults should reset only on explicit reset actions (for example `Reset All`)
- keep input field widths consistent within each group (same width token)
- implement consistent field sizing with a fixed label column and a shared expandable input column so label length does not shrink input controls
- compare parameter dropdown should expose shared parameters and active policy-specific parameters; avoid stale/unused controls
- add short hover tooltips for training-relevant parameters; each tooltip should explain the practical effect on learning speed, stability, exploration, or compute cost in one sentence

#### Live Plot Group Fields
- `Moving average values` (default `20`)
- do not add extra advanced controls for capture count; replay capture density is controlled by `Frame stride`

### Controls Row
Exactly 8 equal-width controls in this order:
1. `Run single episode`
2. `Train and Run`
3. `Pause` / `Run`
4. `Reset All`
5. `Clear Plot`
6. `Save samplings CSV`
7. `Save Plot PNG`
8. `Device` selector (`CPU` / `GPU`, default `CPU`)

Control Highlight Behavior:
- highlight `Train and Run` only while training is active
- while paused, remove `Train and Run` highlight and highlight `Pause/Run` only
- pressing `Train and Run` while paused must cancel the paused run and start a fresh run with current parameter values
- highlight `Pause/Run` only while paused
- use neutral/default style when inactive

### Current Run Panel
- `Steps` label + progress bar
- `Episodes` label + progress bar
- place `Episodes` progress row below the `Steps` progress row
- steps progress advances during replay animation playback only
- status line format:
  - `Epsilon: <...> | LR: <...> | Best reward: <...> | Render: <off|on|skipped|idle>`

### Live Update Strictness
- provide a strict visual update mode for debugging/analysis where event pump batch size is `1` and pump interval is around `16 ms`
- in strict mode, force immediate plot repaint for primary per-episode updates (`draw`) instead of relying only on idle coalescing
- default mode may batch events for throughput, but strict mode must support visibly incremental episode-by-episode progression

### Live Plot Panel
- no title
- axis labels: `x = Episodes`, `y = Reward`
- render per-episode reward with slight transparency (recommended `alpha ~ 0.60`)
- use the same run color for per-episode reward, `MA`, and `eval`
- draw `MA` and `eval` with `2x` line width of reward and distinct styles
- use `alpha = 1.0` for both `MA` and `eval` lines
  - recommended: `MA` dashed (`--`), `eval` dotted (`:`) with marker
- legend outside right side
- show full base label once per run, compact `moving average` / `evaluation rollout` entries
- use plain legend labels `moving average` and `evaluation rollout` (no run-id suffix)
- support explicit line breaks in reward legend text to prevent cutoff
- reward legend text format:
  - line 1: `policy | steps=<max_steps> | gamma=<gamma>`
  - line 2: `epsilon=<...> | epsilon_decay=<...> | epsilon_min=<...>`
  - line 3: `LR=<...> | LR strategy=<...> | LR decay=<...>`
  - line 4: environment-specific params (`key=value | ...`)
- in compare mode, append compared parameter key/value pairs to each run legend entry when they are not already represented in the base legend lines
- avoid duplicate fields in compare legend enrichment (if a field is already shown in the base legend, do not repeat it)
- legend interactive toggling (text + handle clickable)
- legend hover affordance: on hover, legend entry appearance changes and cursor switches to hand to indicate clickability
- when legend content exceeds plot height, support legend scrolling via mouse wheel while hovering the legend
- legend run-visibility toggles should persist across subsequent plot redraws/new episode updates (do not auto-reenable hidden runs)
- reserve right gutter for legend and keep plot left-aligned
- subplot margins reserve legend gutter (target: `left~0.04`, `right~0.78`)
- preserve run history until `Clear Plot`
- legend/run labels must use immutable per-run parameter snapshots captured at run start (do not read live control values for historical runs)
- grid/spine/tick/label/legend text colors aligned with GUI text tone
- live plot grid must be enabled by default in runtime configuration (not only styled)

---

## Styling Baseline (Dark Mode)
- theme preference: `clam`, fallback `vista`
- palette:
  - main bg `#1e1e1e`
  - panel bg `#252526`
  - input bg `#2d2d30`
  - text `#e6e6e6`
  - muted `#d0d0d0`
  - accent `#0e639c`
- fonts:
  - default: `Segoe UI`, size `10`
  - group headings: `Segoe UI`, size `10`, bold
  - control buttons: `Segoe UI`, size `10`, bold
- spacing tokens:
  - outer pad `10`
  - inner pad `6`
  - tight pad `4`
  - label column min width `~92`
  - parameter input width `~9`
- button styles:
  - neutral: bg `#3a3d41`, active `#4a4f55`, pressed `#2f3338`
  - train-accent: `#0e639c`, active `#1177bb`, pressed `#0b4f7a`
  - pause-amber: `#a66a00`, active `#bf7a00`, pressed `#8c5900`
- apply explicit neutral style to inactive action buttons (do not fall back to platform default button style)
- combobox listbox:
  - bg `#2d2d30`, fg `#e6e6e6`, selected bg `#0e639c`, selected fg `white`
- text insertion cursor in entry widgets should be white for visibility on dark background
- entry/combobox field visuals should be explicitly styled to dark-mode tokens (avoid platform-default light fields)
- disable mousewheel value changes for dropdown/combobox controls (mousewheel should not cycle selected option)
- progressbar:
  - trough `#343434`, fill `#0e639c`

---

## GUI Robustness Tests
- GUI smoke tests: startup, clear/reset safety, plotting/legend interaction
- policy-switch regression: changing policy updates visible policy-specific controls/options without overwriting current entered values
- policy-isolation regression: all `Specific` group parameters (including `gamma`, `learning_rate`, `batch_size`, `hidden_layer`, `lr_strategy`, `min_lr`, `lr_decay`) remain independent per policy across policy switches
- selector regression: `activation` and `lr_strategy` controls are readonly selectors with expected option sets
- specific-layout regression: shared rows are above separator and policy-specific rows are below separator
- reset regression: `Reset All` restores general defaults and active-policy defaults consistently
- compare finalization consistency
- compare render regression: with compare enabled, only the selected render run updates live animation; other runs still update plot/statistics
- compare input regression: `Tab` autofill places caret at end of inserted value and preview hint remains aligned with value-input column
- combobox interaction regression: mousewheel does not cycle dropdown selections
- pause/restart regression: while paused, `Train and Run` is neutral and pressing it starts a fresh run from current parameters
- stale-event regression: ignore worker events with non-current run/session identifiers so old runs cannot overwrite active UI state
- split-event regression: `episode` updates reward/MA every episode immediately; `episode_aux` can arrive later and must update eval/frames without duplicating reward points
- shutdown regression: on reset/close, paused workers are resumed before stop so the process exits cleanly
- animation runtime hotfix regression:
  - toggling `Animation on = False` clears queued playback frames immediately
  - replay progress reset
  - status render state updates to `off`
- animation payload regression: when episode events include multiple frames, GUI replays full sequence at configured FPS (not only latest frame)
- animation queue regression: while playback is active, new episode payloads should not interrupt current playback; only the newest pending playback should run next
- legend overflow regression: when legend height exceeds plot panel height, mousewheel scrolling while hovering legend moves legend content within bounded range
- legend persistence regression: if a run is hidden via legend toggle, it remains hidden after subsequent plot redraws/episode updates until explicitly re-enabled
- compare GPU-fallback regression: with `Device = GPU` and CUDA unavailable, compare mode remains stable using CPU-effective execution path
- if Tk/Tcl assets are unavailable, skip with explicit reason

---

## Benchmark-Derived Performance Guidance
Use measured behavior to prioritize optimization work:
- gradient update path is the primary runtime bottleneck (roughly `~2x` cost vs no-gradient diagnostic runs)
- GUI overhead is small (single-digit percent range in measured setup)
- animation overhead is secondary compared to gradient updates for this workload

Blueprint implications:
- keep UI/render optimizations focused on responsiveness, not as the main throughput lever
- for throughput tuning, prioritize backend update cadence and model compute controls first (`train_freq`, `gradient_steps`, network size, precision)
- preserve easy switching between:
  - analysis mode: strict per-episode visualization
  - throughput mode: coalesced plotting and reduced UI work
