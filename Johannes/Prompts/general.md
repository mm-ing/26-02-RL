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

---

## Project Output Structure
Create (using `<project_name>` from the project-specific file):
- `<project_name>_app.py`
- `<project_name>_logic.py`
- `<project_name>_gui.py`
- `tests/test_<project_name>_logic.py`
- `tests/test_<project_name>_gui.py`
- `requirements.txt`
- `README.md`
- output folders: `results_csv/`, `plots/`

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
