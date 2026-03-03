# Prompt Assembly Guide

Use this folder to assemble one generation prompt from shared rules + one project-specific file.

## Required files
- `general.md`
- `logic.md`
- `gui.md`
- one project-specific markdown (for example `InvDoubPend.md`)

## Strict assembly order (copy-paste)
1. Paste `general.md`
2. Paste `logic.md`
3. Paste `gui.md`
4. Paste the project-specific file

Keep this order unchanged.

## Project-specific file must include only
1. `project_name`
2. environment details
3. policy list to expose
4. explicit statement to use `Stable-Baselines3 (SB3)`

## Minimal prompt wrapper template
Use this wrapper around pasted content:

```text
Generate the project using the following specifications.
Follow all constraints exactly.

[PASTE general.md]

[PASTE logic.md]

[PASTE gui.md]

[PASTE project-specific file]
```

## Scope reminder
- Shared behavior and architecture come from `general.md`, `logic.md`, `gui.md`.
- Environment/policy identity comes only from the project-specific file.

## Current GUI sync notes (important)
When maintaining `gui.md`, keep these implementation-critical rules aligned:
- Compare top row: `Compare on` on the left, `Clear` + `Add` on the right.
- Compare input row: side-by-side `Parameter` (left) and `Values` (right), no extra field labels above inputs.
- Compare completion hint: shown directly below the `Values` input (for example `Tab -> Tanh`).
- Compare interaction: pressing `Enter` in the `Values` input triggers `Add`.
- Live plot layout: reserve right legend gutter with subplot targets `left~0.04`, `right~0.78`.
- Controls row device behavior: include a `Device` selector with `CPU`/`GPU`, default `CPU`, and safe fallback to `CPU` when CUDA is unavailable.
- Session/event robustness: stale events from previous runs must be ignored via run/session identifiers.
