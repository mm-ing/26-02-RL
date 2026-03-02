# Parameters

This file documents all GUI parameters used by the LunarLander app and how they map to current implementation behavior.

---

## 1) Environment

### Current device (button)
- Label toggles between `Current device: CPU` and `Current device: GPU`.
- Default is CPU.
- Switching is blocked while training is running.
- If GPU is requested but CUDA is unavailable, app falls back to CPU.
- Device change clears active agents, so new runs are created on the selected device.

### Animation on
- Enables/disables GUI rendering updates only.
- Does not change training logic.

### Animation FPS
- Controls render refresh frequency.
- Higher values improve visual smoothness but increase GUI workload.

### Continuous (`continous` GUI field)
- Sets environment action mode.
- Discrete policies force `continuous=False`.
- Continuous policies force `continuous=True`.
- In compare mode this is auto-adjusted per run based on policy.

### Gravity / Wind on / Wind power / Turbulence power
- Passed to environment rebuild.
- Affects task dynamics for all policies.

### Update (button)
- Rebuilds environment from current environment controls.

---

## 2) Compare Mode

### Compare on
- Enables multi-run compare execution.
- Workers are launched per parameter-combination run.

### Parameter + Options
- `Parameter` chooses what to vary.
- `Options` accepts comma-separated values for that parameter.
- Values are parsed by parameter type (int/float/bool/enum/string).

### Compare summary
- Shows active compare lists currently stored in GUI state.

### Clear
- Clears all compare lists and summary.

### Combination rule
- Runs are generated as Cartesian product of all parameter lists.
- If list sizes are $n_1, n_2, ..., n_k$, number of runs is:

$$
N_{runs} = \prod_{i=1}^{k} n_i
$$

### Policy-specific default behavior (important)
- If `Policy` is in compare parameters, each run first applies that policy’s default hyperparameters.
- Then only explicitly compared parameters override those defaults.
- This avoids accidental cross-policy bias from the base GUI snapshot.

---

## 3) General

### Policy
- Selects algorithm family.
- Discrete: `DuelingDQN`, `D3QN`, `DDQN+PER`.
- Continuous: `PPO`, `A2C`, `TRPO`, `SAC`.

### Max steps
- Max env steps per episode.

### Episodes
- Number of training episodes in Train-and-Run.

### Epsilon max / Epsilon decay / Epsilon min
- Used directly by discrete epsilon-greedy exploration.
- Continuous policies do not rely on epsilon for action selection.
- Decay rule:

$$
\epsilon_{t+1} = \max(\epsilon_{min},\epsilon_t\cdot d)
$$

---

## 4) Specific Hyperparameters

### Gamma
- Discount factor for returns/targets.
- Used by all policies.

### Learning rate
- Base optimizer LR.
- Can be modified by selected LR strategy.

### Replay size
- Replay capacity for off-policy training.
- Used by discrete policies and SAC.

### Batch size
- Minibatch size for updates.
- Used by discrete and SAC replay updates.
- Also used in continuous on-policy minibatch slicing.

### Target update
- Discrete: hard target sync cadence.
- SAC: soft target critic updates.
- Not used by PPO/A2C/TRPO.

### Replay warmup
- Minimum replay size before learning starts.
- Used by discrete and SAC.

### Learning cadence
- Discrete/SAC: step-gated replay update cadence.
- PPO/A2C/TRPO: rollout update cadence plus episode-end flush.

### Activation
- Hidden-layer activation function for networks.

### Hidden layers
- Comma-separated hidden layer sizes, e.g. `256,256`.

### LR strategy
- `exponential`, `linear`, `cosine`, `loss-based`, `guarded natural gradient`.
- Applied in both discrete and continuous agents.

### LR decay
- Defines target LR ratio and strategy-specific decay strength.

### Min LR
- Lower bound clamp for effective learning rate.

### GAE λ
- GAE parameter for continuous trajectory advantage estimation.

$$
\hat A_t^{GAE(\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

### PPO clip
- PPO ratio clipping range.
- Used only by PPO.

$$
L^{PPO}_{clip} = -\min(r_t\hat A_t, \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)
$$

---

## 5) Live Plot

### Moving average values
- Window size for moving-average reward line.

### Deterministic eval tracking
- In Train-and-Run and Compare mode, periodic deterministic evaluation episodes are recorded.
- Eval is side-effect free: no training updates, no replay insertions, and no optimizer steps from eval rollouts.
- Plot contains per-run training reward/MA and separate eval curve + legend entry (`<base label> | eval`).

---

## 6) Policy Behavior Summary

### Discrete (`DuelingDQN`, `D3QN`, `DDQN+PER`)
- Epsilon parameters are active.
- Replay warmup/cadence/target update are central.
- `DDQN+PER` adds prioritized replay.

### Continuous (`PPO`, `A2C`, `TRPO`, `SAC`)
- Action selection is policy-distribution based.
- PPO uses clipped objective.
- A2C uses on-policy actor-critic objective.
- TRPO uses KL-constrained trust-region step (CG + line search).
- SAC uses off-policy replay with twin critics, target critics, and entropy temperature.

---

## 7) Practical Tuning

- Start with policy-default settings first.
- Sweep one parameter at a time before multi-parameter compare.
- Keep compare list sizes moderate to control Cartesian growth.
- For policy comparisons, prioritize deterministic eval curves over raw train reward noise.
