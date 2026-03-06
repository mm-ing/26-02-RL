# General Prompt Blueprint

## Purpose
Use this file for shared, project-agnostic generation rules.
Combine it with:
- `logic.md` (architecture + training logic rules)
- `gui.md` (layout/style/runtime GUI behavior)
- one **project-specific** markdown file (minimal File B)

Project-specific file must contain only:
1. project name
2. environment details
3. policy list to expose
4. explicit statement to use `Stable-Baselines3 (SB3)`

---

## File Responsibilities
- `general.md`: contract, structure, tech/backend rules, reproducibility
- `logic.md`: architecture, policy/training/runtime logic, compare execution, logic tests
- `gui.md`: layout, controls/inputs/progress/plot behavior, style, GUI threading rules

Use this order:
1. read `general.md`
2. apply project-specific file
3. implement `logic.md`
4. implement `gui.md`

---

## Two-File Contract
- File A (`general.md` + `logic.md` + `gui.md`): fixed shared architecture and behavior
- File B (project-specific): environment + policies + SB3 requirement

If File B omits optional values, use master defaults.

---

## Technology and Backend Requirements
- Language: Python 3.8+
- GUI: Tkinter + ttk
- Plotting: matplotlib (TkAgg)
- Image rendering: Pillow (if available)
- RL backend: **Stable-Baselines3 / sb3-contrib only**
- Deep learning runtime: PyTorch (as used by SB3)
- Environment API: gymnasium

Do not implement non-SB3 custom algorithms when SB3 is required.

Implementation guardrails:
- do not rely on private matplotlib internals (for example `_get_lines.prop_cycler`); use stable/public APIs (for example `matplotlib.rcParams['axes.prop_cycle']`)
- in worker-threaded training, never pass GUI/session-only keys into strict trainer config dataclasses unless those fields are explicitly declared
- when multiple worker threads share mutable GUI-side worker registries, protect registry mutation/read with a lock to avoid race conditions in pause/cancel paths
- keep GUI animation replay decoupled from training progress; replay scheduling must never block or restart worker training loops
- preserve immutable per-run metadata snapshots for persisted UI artifacts (for example legend labels) so later control edits cannot overwrite historical run descriptions
- for user-facing parameter hints/tooltips, keep copy concise and effect-oriented (what changes in training dynamics or compute), avoiding algorithm-specific jargon when not required

### Animation Performance Mode (Default)
For training-time animation in GUI projects, use this default behavior unless a project-specific file explicitly overrides it:
- capture animation frames during the training loop via callback/event hooks (reuse training steps)
- avoid separate post-episode rollout runs solely for animation
- use `Update rate (episodes)` to select which episodes emit animation buffers
- use `Frame stride` to control per-episode frame sampling density
- playback must be non-blocking relative to worker training loops
- when playback is active, keep only one pending animation buffer and apply newest-buffer overwrite (`latest-wins`)

---

## Project Output Structure
Create (using `<project_name>` from the project-specific file):
- `<project_name>_app.py`
- `<project_name>_logic.py`
- `<project_name>_gui.py`
- `<project_name>_REQUIREMENTS_MATRIX.md`
- `tests/test_<project_name>_logic.py`
- `tests/test_<project_name>_gui.py`
- `requirements.txt`
- `README.md`
- output folders: `results_csv/`, `plots/`

---

## Requirements Matrix Governance (Required)
For every generated project, create and maintain `<project_name>_REQUIREMENTS_MATRIX.md`.

Minimum matrix structure:
- `GUI Contract Coverage` table (`Area`, `Requirement`, `Status`, `Notes`)
- `Logic Contract Coverage` table (`Area`, `Requirement`, `Status`, `Notes`)
- `Test Coverage` table (`Area`, `Status`, `Notes`)
- `High-Impact Open Gaps` section

Status values:
- `✅ Implemented`
- `🟡 Partial`
- `❌ Missing`

Process rules:
- initialize matrix at project bootstrap
- update matrix after each implementation/testing pass
- before handoff, high-impact blockers must be resolved or explicitly justified
- matrix must reflect latest isolated test run outcome

### Mandatory Final Contract Recheck (All Future Projects)
Before handoff, always run a final contract recheck against:
1. `general.md`
2. `logic.md`
3. `gui.md`
4. the project-specific file

Recheck process requirements:
- verify each contractual requirement as `✅ Implemented`, `🟡 Partial`, or `❌ Missing`
- ensure no matrix row is marked `✅ Implemented` unless the code and tests both support it
- explicitly list any remaining high-impact gaps with concrete remediation notes
- update the requirements matrix in the project directory to reflect this final audit

---

## App Module Baseline
- Provide a simple entrypoint: create `Tk()`, instantiate GUI, start `mainloop()`
- Set startup env guards **before importing GUI/ML modules**:
  - `TF_ENABLE_ONEDNN_OPTS=0`
  - `TF_CPP_MIN_LOG_LEVEL=3`
- Apply MuJoCo GL startup guard:
  - on Windows: if `MUJOCO_GL=angle` is pre-set, clear it (to avoid invalid backend startup failure)
  - on non-Windows: default `MUJOCO_GL=egl` if not already set

---

## Requirements Baseline
Include at least:
- `stable-baselines3`
- `sb3-contrib`
- `torch`
- `matplotlib`
- `pillow`
- `pytest`

---

## Reproducibility Rule
If the project-specific file provides only:
- project name
- environment details
- policy list
- `use SB3`

Ensure these split master files are sufficient to regenerate a functionally equivalent project.
