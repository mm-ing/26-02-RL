# Project Comparison: Johannes `LunarLander` vs Manfred `Lunar_Lander`

## Scope
This comparison focuses on factors that influence:
1. **training quality/results** (final policy performance, stability),
2. **runtime performance** (steps/sec, wall-clock speed, responsiveness).

Compared projects:
- `Johannes/LunarLander` (custom RL implementations)
- `Manfred/Lunar_Lander` (Stable-Baselines3-based workbench)

---

## Executive summary
Manfred’s project can reasonably appear “better” because it relies on **mature, highly tuned SB3 implementations** and a **train-then-evaluate workflow** that often reports cleaner returns. Johannes’ project is more research/flexible and now includes true continuous implementations, but still has more custom moving parts (optimizers, schedulers, cadence semantics, GUI coupling), which can increase variance and reduce throughput.

Most likely high-impact differences are:
1. **Algorithm backend quality**: SB3 core loops vs custom loops.
2. **Evaluation protocol**: deterministic eval episodes (Manfred) vs reward from ongoing training episodes (Johannes).
3. **Compute overhead pattern**: bulk `model.learn(...)` chunks (Manfred) vs per-step Python updates + GUI/live plotting (Johannes).
4. **Hyperparameter semantics mismatch** between custom and SB3 defaults.

---

## Main differences and expected impact

## 1) Algorithm implementation strategy

### Manfred (`lunar_lander_logic.py`)
- Uses `stable_baselines3` (`DQN`, `PPO`, `A2C`, `SAC`) and optional `sb3_contrib.TRPO`.
- Most training is delegated to `model.learn(total_timesteps=...)`.
- Device set to `device="auto"`.

### Johannes (`LunarLander_logic.py`)
- Implements custom agents and update loops directly in PyTorch.
- Includes custom implementations for:
  - DQN variants + PER,
  - PPO, A2C, TRPO (trust-region step), SAC (twin critics + alpha).

### Impact
- **Training results**: SB3 typically gives more predictable baseline quality out of the box.
- **Performance**: SB3’s internals are heavily optimized; custom Python-heavy loops can be slower, especially when update frequency is high.

---

## 2) Training/evaluation protocol

### Manfred
- `TrainLoop.run_episode()` does:
  1. train chunk via `algorithm.update({...})` (`model.learn(...)`),
  2. evaluate policy separately (`evaluate_episode`) at intervals,
  3. non-eval episodes may use quick monitor-based metric.
- Evaluation uses deterministic policy action (`predict(..., deterministic=True)`).

### Johannes
- `Trainer.run_episode()` interleaves acting/storing/learning **inside the same episode**.
- Reward curves are from training-time trajectories (with exploration/noise effects).

### Impact
- **Training results perception**: Manfred often shows “cleaner/better” curves because evaluation is partially decoupled from noisy training actions.
- **Fair comparison risk**: raw return curves are not directly equivalent unless both use the same evaluation protocol.

---

## 3) Throughput and UI overhead

### Manfred
- Training loop primarily calls chunked `model.learn`.
- Rendering is not done every episode step in training; evaluation rendering is interval-based.
- Workbench is multi-job oriented with event bus and status updates.

### Johannes
- Frequent per-step operations during training episodes:
  - action, env step, store, learn-step calls,
  - progress callbacks,
  - live plot updates and GUI status updates,
  - optional rendering pipeline.
- Compare mode can run many parameter combinations and plotting updates.

### Impact
- **Runtime performance**: Johannes can lose throughput due to Python-level per-step overhead and GUI/plot cadence work.
- **User-visible speed**: Manfred may feel faster for equivalent timesteps because more work is batched.

---

## 4) Hyperparameter and scheduler semantics

### Manfred
- SB3-specific knobs: `train_freq`, `gradient_steps`, `learning_starts`, etc.
- LR schedules mapped to SB3 learning-rate callable (`Constant/Linear/Exponential/Step`).

### Johannes
- Custom scheduler system (`exponential`, `linear`, `cosine`, `loss-based`, `guarded natural gradient`).
- `learning_cadence`, `replay_warmup`, and per-policy training-plan mapping are custom semantics.

### Impact
- **Training results**: same numeric values can behave differently across projects.
- **Tuning complexity**: Johannes requires policy-specific interpretation of cadence/scheduler interactions.

---

## 5) Continuous control maturity

### Manfred
- Uses SB3 PPO/A2C/SAC and contrib TRPO implementations (well-tested algorithm internals).

### Johannes
- Recently upgraded to true continuous paths:
  - A2C dedicated actor-critic objective,
  - TRPO trust-region CG + line search,
  - SAC off-policy replay + twin critics + alpha.

### Impact
- **Potential upside**: Johannes now has strong algorithmic flexibility and transparency.
- **Potential downside**: custom implementations can still lag SB3 in numerical robustness/performance tuning without extended benchmarking.

---

## 6) Reproducibility and benchmarking practice

### Manfred
- Tests include simulation checks where baseline vs trained return is compared.
- Explicit random seed usage appears in training tests (`set_random_seed(42)`).

### Johannes
- Strong unit coverage for logic/GUI behavior and regressions.
- Less emphasis (currently) on standardized multi-seed eval benchmark as first-class artifact.

### Impact
- **Results confidence**: Manfred pipeline may produce more reproducible “headline” improvement checks.
- **Diagnosis speed**: Johannes has better behavioral regression safety, but needs stronger benchmark protocol for apples-to-apples performance claims.

---

## What most likely explains “Manfred performs better”

Most probable contributors (in descending order):
1. **SB3 algorithm kernels and defaults** are mature and stable.
2. **Evaluation method** in Manfred is cleaner (deterministic eval episodes), while Johannes curves include training noise.
3. **Batch/chunk training calls** in Manfred reduce Python overhead.
4. **Custom scheduler/cadence interactions** in Johannes can under/over-update depending on policy and horizon.
5. **GUI/live plotting overhead** in Johannes can reduce effective training throughput.

---

## Practical recommendations for fair comparison

1. **Unify evaluation protocol**
   - For both projects: report deterministic evaluation returns every fixed interval (same seeds, same episode count, same max steps).

2. **Run matched benchmark matrix**
   - Policies: `DuelingDQN`, `D3QN`, `DDQN+PER`, `PPO`, `A2C`, `TRPO`, `SAC` where supported.
   - 5+ seeds, fixed environment config.
   - Report mean, median, std, and steps/sec.

3. **Separate training speed from UI overhead**
   - Benchmark in headless/minimal-plot mode for Johannes.
   - Compare raw timesteps/sec independent of rendering.

4. **Policy-specific tuning alignment**
   - Map Johannes cadence/warmup/scheduler settings to SB3-equivalent update intensity before judging algorithm quality.

5. **Use identical device policy**
   - CPU-only and GPU-only runs separately (same seeds/configs).

---

## Action plan (high-impact next steps)

1. **Add deterministic eval mode to Johannes trainer**
   - Every `N` episodes, run a no-training evaluation rollout with deterministic actions and log it separately from training returns.
   - Expected impact: cleaner comparability and better signal on true policy quality.

2. **Introduce headless benchmark switch**
   - Disable rendering, plot redraw, and UI status churn during benchmark runs.
   - Expected impact: higher and more stable steps/sec; clearer algorithm-vs-UI performance split.

3. **Batch update option for custom agents**
   - Add optional train chunks (e.g., multiple gradient updates per environment block) to reduce per-step Python overhead.
   - Expected impact: closer runtime characteristics to SB3 `learn(...)` chunks.

4. **Publish fixed benchmark protocol in repo**
   - Standard script/config: seeds, episodes, max steps, env params, device mode, output metrics.
   - Expected impact: reproducible comparisons across commits and across both projects.

5. **Policy-specific calibration pass**
   - Tune Johannes defaults to matched update intensity against SB3 (especially cadence/warmup/gradient frequency analogs).
   - Expected impact: reduce systematic under/over-training artifacts and close quality gap.

---

## Bottom line
- **Manfred project advantage**: production-grade algorithm backend + cleaner evaluation workflow.
- **Johannes project advantage**: full control/inspectability, advanced custom features (compare combinations, custom schedulers, now true continuous paths).
- The current “better performance” signal is likely a mixture of **real algorithm/runtime differences** and **measurement protocol differences**. Standardized evaluation is required for a fair final verdict.
