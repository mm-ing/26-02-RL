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
- publish structured events (`step`, `episode`, `training_done`, `error`)
- tag worker-originated events with `session_id` in the GUI-side worker bridge and ignore stale session events in the pump
- pause/resume/cancel actions must control all active worker trainers for the current session (single-run and compare)
- keep the active worker registry thread-safe (lock reads/writes) and pre-register workers before launching jobs so pause works immediately

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
- for training animation, replay rollout frame sequences at `Animation FPS` (no single-frame-only behavior)
- replay queue semantics: if a playback is active, keep it running and store only one pending playback slot; incoming newer playback replaces pending slot (`latest-wins`)

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
- `Update` button
- environment-specific parameters below `Update`:
  - `healthy_reward`
  - `reset_noise_scale`

#### Compare Group Fields
- `Compare on` toggle
- top control row: `Compare on` at left, `Clear` and `Add` buttons at right (side-by-side)
- compare parameter dropdown (`General` + `Specific` parameters only; exclude `Environment`)
- compare values text input (comma-separated)
- second row layout: `Parameter` dropdown (left) and `Values` input (right), without additional field labels
- compare values suggestions + `Tab` completion for categorical params (`Policy`, `Activation`, `LR strategy`)
- compare values completion preview hint shown below the `Values` input (for example `Tab -> Tanh`)
- summary lines: `Parameter: [v1, v2, ...]`

Rules:
- when compare is on, set animation off automatically (user may re-enable)
- `Add` commits selected parameter + values into active compare lists
- pressing `Enter` in compare values input triggers `Add`
- `Clear` removes all active compare parameter lists
- include all `General` + `Specific` parameters in compare dropdown; exclude `Environment`
- allow accepting categorical suggestion with `Tab` based on typed prefix
- when compare mode runs multiple workers, only one selected run should feed live animation/playback (selected policy preferred; fallback first run)

#### General Group Fields
- `Max steps`
- `Episodes`
- `Epsilon max`
- `Epsilon decay`
- `Epsilon min`
- `Gamma`

#### Specific Group Fields
- top row: `Policy`
- row order:
  1. `Hidden layer` | `Activation`
  2. `LR` | `LR strategy`
  3. `Min LR` | `LR decay`
  4. `Replay size` | `Batch size`
  5. `Learning start` | `Learning frequency`
  6. `Target update`

Notes:
- use scientific notation for LR text inputs
- apply policy defaults on policy change
- keep input field widths consistent within each group (same width token)

#### Live Plot Group Fields
- `Moving average values` (default `20`)
- `Show Advanced` toggle (default `False`)
- advanced fields:
  - `Rollout full-capture steps` (default `120`)
  - `Low-overhead animation` (default `False`)

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
- steps progress advances during replay animation playback only
- status line format:
  - `Epsilon: <...> | LR: <...> | Best reward: <...> | Render: <off|on|skipped|idle>`

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
- legend interactive toggling (text + handle clickable)
- legend hover affordance: on hover, legend entry appearance changes and cursor switches to hand to indicate clickability
- reserve right gutter for legend and keep plot left-aligned
- subplot margins reserve legend gutter (target: `left~0.04`, `right~0.78`)
- preserve run history until `Clear Plot`
- legend/run labels must use immutable per-run parameter snapshots captured at run start (do not read live control values for historical runs)
- grid/spine/tick/label/legend text colors aligned with GUI text tone

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
- combobox listbox:
  - bg `#2d2d30`, fg `#e6e6e6`, selected bg `#0e639c`, selected fg `white`
- progressbar:
  - trough `#343434`, fill `#0e639c`

---

## GUI Robustness Tests
- GUI smoke tests: startup, clear/reset safety, plotting/legend interaction
- compare finalization consistency
- compare render regression: with compare enabled, only the selected render run updates live animation; other runs still update plot/statistics
- pause/restart regression: while paused, `Train and Run` is neutral and pressing it starts a fresh run from current parameters
- stale-event regression: ignore worker events with non-current run/session identifiers so old runs cannot overwrite active UI state
- shutdown regression: on reset/close, paused workers are resumed before stop so the process exits cleanly
- animation runtime hotfix regression:
  - toggling `Animation on = False` clears queued playback frames immediately
  - replay progress reset
  - status render state updates to `off`
- animation payload regression: when episode events include multiple frames, GUI replays full sequence at configured FPS (not only latest frame)
- animation queue regression: while playback is active, new episode payloads should not interrupt current playback; only the newest pending playback should run next
- if Tk/Tcl assets are unavailable, skip with explicit reason
