# Executable Prompt: Add Compare Mode to LunarLander

You are GitHub Copilot (GPT-5.3-Codex) working only in:

- `26-02-RL/Johannes/LunarLander/`

Implement a new Compare Mode in the existing project (do not rebuild from scratch). Keep existing behavior unchanged when compare mode is off.

## Goal
Add a compare capability so that when enabled, pressing `Train and Run` launches parallel training runs for all available policies, while live environment animation remains tied to the currently selected policy in the policy dropdown.

## Current project assumptions
- Available policies currently are:
  - `DuelingDQN`
  - `D3QN`
  - `DDQN+PER`
- `DoubleDQN` has already been removed.
- Existing GUI/threading/plot behavior is implemented and should be preserved unless explicitly changed below.

## Files to modify
- `LunarLander_gui.py`
- `LunarLander_logic.py` (only if minimal helper additions are needed)
- `README.md` (document compare mode)
- `tests/test_LunarLander_logic.py` (only if logic-level tests are needed)

Prefer minimal focused edits.

## Functional requirements

### 1) Parameters panel: new Compare group
In the `Parameters` panel, insert a new group between `Environment` and `General` named:
- `Compare`

Inside it add:
- Toggle input: `Compare on` (default: off)

Implementation detail:
- Use a Tk variable (e.g., `BooleanVar`) with default `False`.
- Ensure layout and spacing match existing dark-mode/styled GUI.

### 2) Train and Run behavior
When `Compare on` is off:
- Keep existing behavior exactly.

When `Compare on` is on and user clicks `Train and Run`:
- Start parallel training runs for all available policies.
- Use the same general training settings (episodes, max steps, epsilon schedule, etc.) from the current UI snapshot.
- Apply each policy’s own specific defaults/parameters from internal policy config storage.
- Collect rewards for each policy independently.
- Each policy run must be represented in live plot as its own run (reward + moving average entries with existing legend format/pairing).
- Keep UI responsive and thread-safe.

### 3) Environment animation rule in compare mode
When compare mode is on:
- Environment animation should be live only for the policy currently selected in the policy dropdown.
- Other policies train in parallel but do not drive the render canvas.

Implementation hint:
- Either:
  - Use separate trainer/env instances per policy worker and route render updates from selected policy only, or
  - Keep one render-designated worker (selected policy) and run non-render workers without UI rendering.
- Ensure no cross-thread Tk updates.

### 4) Ploting in compare mode
- Continue using current live plot style, interactivity, and legend behavior.
- For compare mode, append one run per policy for each compare launch.
- Preserve existing legend format:
  - `<policy> | eps(<epsilon max>/<epsilon min>) | lr=<learning rate> | reward`
  - `<policy> | eps(<epsilon max>/<epsilon min>) | lr=<learning rate> | MA`
- Keep interactive legend toggles (text and line click).

### 5) Progress and status behavior
- Keep progress bars functional without UI freezes.
- Define deterministic behavior for compare mode progress display:
  - Steps bar reflects selected policy worker progress.
  - Episodes bar can reflect selected policy progress or aggregate completed episodes; choose one and document it in README.
- Status line (`Epsilon | Current x | Best x`) should reflect selected policy worker.

### 6) Pause / Run, Reset All, Clear Plot safety
- Existing safety guarantees must remain valid in compare mode.
- `Pause/Run` should pause/resume compare workers consistently.
- `Reset All` should stop all compare workers safely.
- `Clear Plot` should be safe during compare training.

### 7) Threading and synchronization constraints
- No worker thread may directly update Tk widgets.
- No worker thread may read Tk variables directly after start.
- Snapshot all needed UI values on main thread before launching workers.
- Use shared pending state with locking and coalesced UI pump updates.
- Avoid race conditions between multiple workers writing pending state:
  - Use per-policy keys/structures (do not overwrite across policies).

### 8) Backward compatibility
- Existing single-policy training mode must behave exactly as before.
- Existing dark mode styling should remain intact.
- Existing CSV/PNG export behavior must continue to work.

## Suggested implementation approach
1. Add compare toggle variable and UI group.
2. Add compare branch in `Train and Run` flow.
3. For compare mode, create one worker thread per policy.
4. Use a per-policy pending state structure:
   - `pending[policy] = {episode, step, rewards_snapshot, epsilon, current_x, best_x, finished}`
5. UI pump merges all policies:
   - Update selected-policy status/progress bars.
   - Throttle plot redraw and draw all policy snapshots.
6. Finalize completed policy runs independently into existing run list.
7. Ensure pause/stop events are respected by all workers.

## Acceptance criteria
- New `Compare` group exists between `Environment` and `General` with `Compare on` toggle default off.
- Compare off: behavior unchanged.
- Compare on + `Train and Run`: all available policies train in parallel.
- Live plot shows runs for all policies.
- Render canvas updates only from selected policy.
- App remains responsive; no thread-related Tk exceptions.
- `Pause/Run`, `Reset All`, `Clear Plot` remain safe in compare mode.
- Existing tests pass; add/adjust tests if needed.
- README updated with compare mode usage and progress-display semantics.

## Validation steps
After implementation:
1. Run:
   - `pytest -q tests/test_LunarLander_logic.py`
2. Manual GUI checks:
   - Compare off single-policy run works.
   - Compare on parallel run starts all policies.
   - Switching selected policy affects which environment animation is shown.
   - Legend toggles still work after compare runs.
   - Pause/Run, Reset All, Clear Plot remain stable in compare mode.

## Non-goals
- Do not add new pages/windows.
- Do not redesign layout beyond adding the required `Compare` group.
- Do not change unrelated algorithms or file structure.
