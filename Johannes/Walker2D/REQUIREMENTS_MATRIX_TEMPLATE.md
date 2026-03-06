# <ProjectName> Requirements Matrix

Scope audited:
- `<ProjectName>_gui.py`
- `<ProjectName>_logic.py`
- `tests/test_<ProjectName>_gui.py`
- `tests/test_<ProjectName>_logic.py`

Status legend:
- ✅ Implemented
- 🟡 Partial
- ❌ Missing

## GUI Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---:|---|
| Layout | Top-level panel structure and ratios |  |  |
| Thread bridge | Queue + after-poll + event filtering |  |  |
| Controls | Required controls, order, highlight behavior |  |  |
| Parameters | Required groups/fields/order |  |  |
| Compare | Add/clear, cartesian behavior, selected render run |  |  |
| Plot | Reward/MA/eval styling and legend behavior |  |  |
| Robustness | Animation queue, stale events, shutdown handling |  |  |

## Logic Contract Coverage

| Area | Requirement | Status | Notes |
|---|---|---:|---|
| Backend | SB3-only and exposed policies |  |  |
| Env config | Runtime configurable environment args |  |  |
| Trainer API | run_episode/train/evaluate/update |  |  |
| Events | step/episode/training_done/error payload contract |  |  |
| Runtime | pause/resume/cancel/device fallback |  |  |
| Compare | max concurrency and per-run IDs |  |  |
| Exports | CSV and PNG behavior |  |  |

## Test Coverage vs Prompted Robustness Tests

| Area | Status | Notes |
|---|---:|---|
| Logic smoke tests |  |  |
| GUI smoke tests |  |  |
| Policy-switch regression |  |  |
| Pause/restart regression |  |  |
| Stale-event regression |  |  |
| Compare finalization/render regression |  |  |
| Shutdown regression |  |  |

## High-Impact Open Gaps

1. 
2. 
3. 

## Quality Gate (Before Handoff)

- [ ] All high-impact items are resolved or explicitly justified.
- [ ] Isolated tests pass (`python -m pytest -q --rootdir . --confcutdir . tests`).
- [ ] Matrix reflects latest implementation and tests.
