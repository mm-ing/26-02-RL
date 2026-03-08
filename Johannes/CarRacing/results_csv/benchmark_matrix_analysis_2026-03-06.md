# CarRacing Benchmark Matrix Analysis (2026-03-06)

## Scope and Method
- Environment: CarRacing-v3
- Policy: SAC
- Episode length: 1 episode, 1000 steps
- Protocol: 1 warmup run discarded, then 5 measured runs
- Matrix factors:
  - GUI: on/off
  - Animation: on/off
  - Gradient update: on/off (`learning_starts=1` vs `learning_starts=10_000_000`)
- Runner: `_tmp_benchmark_single_combo.py` (one combo per process)

## Raw Results (mean +/- std, seconds)
| GUI | Animation | Gradient Update | Mean (s) | Std (s) | Source |
|---|---|---|---:|---:|---|
| on | on | on | 26.373 | 0.232 | `benchmark_gui-on_anim-on_grad-on_warmup1_runs5.json` |
| on | on | off | 13.454 | 0.213 | `benchmark_gui-on_anim-on_grad-off_warmup1_runs5.json` |
| on | off | on | 27.363 | 0.829 | `benchmark_gui-on_anim-off_grad-on_warmup1_runs5.json` |
| on | off | off | 13.372 | 0.155 | `benchmark_gui-on_anim-off_grad-off_warmup1_runs5.json` |
| off | on | on | 26.632 | 0.209 | `benchmark_gui-off_anim-on_grad-on_warmup1_runs5.json` |
| off | on | off | 12.724 | 0.098 | `benchmark_gui-off_anim-on_grad-off_warmup1_runs5.json` |
| off | off | on | 25.595 | 0.137 | `benchmark_gui-off_anim-off_grad-on_warmup1_runs5.json` |
| off | off | off | 12.733 | 0.189 | `benchmark_gui-off_anim-off_grad-off_warmup1_runs5.json` |

## Main Findings
1. Gradient update dominates runtime.
- Average with grad on: 26.491 s
- Average with grad off: 13.071 s
- Speedup when disabling updates: **2.027x**

2. Animation is not a major bottleneck in this setup.
- Average animation on: 19.796 s
- Average animation off: 19.766 s
- Speedup from animation off: **1.002x** (effectively zero)

3. GUI overhead is small.
- Average GUI on: 20.140 s
- Average GUI off: 19.421 s
- Speedup from GUI off: **1.037x** (~3.7%)

4. End-to-end spread across all combos.
- Best combo: `gui=off, anim=on, grad=off` at 12.724 s
- Worst combo: `gui=on, anim=off, grad=on` at 27.363 s
- Worst/best factor: **2.151x**

## Interpretation
- The wall-time cost is mainly the training update path (backprop, replay sampling, optimizer step).
- Rendering and GUI processing are second-order effects for this workload.
- For practical optimization, prioritize update cadence/model compute before UI/render changes.

## Speed-Up Recommendations and Expected Factors
### A. Low risk, immediate
1. Keep GUI off for bulk experiments.
- Expected: **1.02x to 1.05x**
- Rationale: measured small GUI overhead.

2. Keep animation off during throughput runs.
- Expected: **~1.00x to 1.03x** in this exact setup
- Rationale: measured impact is near zero here; still useful to reduce noise/variance in longer runs.

### B. Medium impact, moderate risk
1. Reduce update frequency per environment step.
- Example: increase `train_freq` from 1 to 2/4 while keeping `gradient_steps=1`.
- Expected: **1.2x to 1.8x** speedup (depends on reward-quality tolerance).
- Rationale: fewer optimizer calls, same rollout amount.

2. Shrink policy/value network size.
- Example: `256,256` -> `128,128`.
- Expected: **1.1x to 1.3x**
- Rationale: less compute per forward/backward pass.

3. Mixed precision (if numerically stable on your GPU path).
- Expected: **1.1x to 1.4x**
- Rationale: faster tensor core math and less memory bandwidth.

### C. High impact but algorithmically changes training behavior
1. Delayed or staged updates.
- Example: warmup longer before first updates, or periodic update windows.
- Expected during early phase: up to **~2.0x**
- Rationale: update path is the dominant cost.

2. Gradient update off for data-collection-only/diagnostic passes.
- Expected: **~2.03x** (measured)
- Rationale: directly removes dominant cost, but no learning occurs.

## Practical Target Estimates
If your baseline is around 26.5 s (grad on), realistic targets are:
- Conservative optimization package (GUI off + minor tuning): **~24 to 25.5 s** (**1.04x to 1.10x**)
- Medium package (reduced update cadence + smaller net): **~15 to 22 s** (**1.2x to 1.75x**)
- Aggressive throughput mode (very sparse updates): **~13 to 16 s** (**1.65x to 2.0x**)

## Suggested Next Benchmark Set
To validate best speed/quality trade-off, run a focused sweep on grad-on only:
1. `train_freq`: 1, 2, 4
2. `gradient_steps`: 1, 2
3. `hidden_layer`: `256,256` vs `128,128`
4. `num_envs`: 2 vs 4

Measure both:
- Throughput (seconds per episode)
- Learning quality proxy (reward after fixed timesteps)

This will quantify true speed-up without optimizing away learning itself.
