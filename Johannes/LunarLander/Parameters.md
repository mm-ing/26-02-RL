# Parameters.md

This document explains every GUI-adjustable parameter in the LunarLander app:
- what it does,
- how it is implemented,
- how behavior depends on the selected policy.

The implementation flow is generally:
1. GUI variables are captured in `_snapshot_ui()` in `LunarLander_gui.py`.
2. Values are applied via `Trainer.set_policy_config(...)` and/or `Trainer.rebuild_environment(...)`.
3. During training, policy-specific agents in `LunarLander_logic.py` consume those values.

---

## 1) Environment Group

## Control-bar device switch

### Current device button
- **Label behavior:** `Current device: CPU` by default; changes to `Current device: GPU` after successful switch.
- **What it does:** Toggles model-device preference between CPU and GPU for newly created agents.
- **Default:** `Current device: CPU`.
- **How implemented:** Button click calls `toggle_device()` -> `set_device_preference(...)`; agent creation uses `get_device()` in logic.
- **Runtime behavior:**
  - Switching is blocked while training is running.
  - If GPU is requested but CUDA is unavailable, it falls back to CPU and shows a warning.
  - Existing agents are cleared when switching device so subsequent runs are created on the selected device.
- **Policy dependency:** None. Applies to all policies.

### Animation on
- **What it does:** Enables/disables environment animation rendering in the GUI.
- **How implemented:** Checked by `_render_tick()`; when off, render updates are skipped.
- **Policy dependency:** None. Visual only.

### Animation FPS
- **What it does:** Sets animation refresh frequency.
- **How implemented:** `_render_tick()` computes interval as `1 / fps`.
- **Policy dependency:** None. Visual only.

### Update (button)
- **What it does:** Rebuilds environment with current environment settings.
- **How implemented:** `update_environment()` -> `Trainer.rebuild_environment(...)`.
- **Policy dependency:** Indirect. Rebuild clears existing agents; policy mode constraints may later re-adjust `continous`.

### continous
- **Policy dependency:** **Strong.**
  - Discrete policies (`DuelingDQN`, `D3QN`, `DDQN+PER`) require `continuous=False`.
  - Continuous (`PPO/A2C`): used for rollout minibatch slicing in policy/value updates.
  - Continuous (`TRPO`): used for critic minibatches; policy step is full-batch with trust-region solve.

### Gravity
- **Policy dependency:** Affects all policies equally via environment dynamics.

### Wind on
- **Formula (discrete hard sync):** if step $t$ satisfies $t \bmod K = 0$ with $K=\text{target\_update}$,
- **Policy dependency:** Affects all policies equally.

### Wind power

SAC soft update each learning step:

$$
	heta^- \leftarrow (1-\tau)\,\theta^- + \tau\,\theta
$$
- **What it does:** Magnitude of horizontal wind force.
- `learning cadence` is used by both paths, but with different semantics: rollout-gated for `PPO/A2C/TRPO`, replay-step-gated for `SAC`.
- `A2C` now uses a dedicated actor-critic on-policy update (advantage-weighted log-prob actor term + value regression).
- `SAC` now uses true off-policy replay + twin target critics + entropy temperature update.
- `TRPO` now uses a KL-constrained trust-region policy step (conjugate gradient + backtracking line search).
- Continuous training uses additional stabilization internals (gradient clipping, NaN guards; entropy terms for PPO/SAC).
- **Core formulas used in implementation:**
  - Advantages (normalized, on-policy path `PPO/A2C/TRPO`):
---

## 2) Compare Group

  - TRPO trust-region constraint (policy step accepted only if satisfied):

$$
\mathbb{E}_t\left[D_{KL}(\pi_{old}(\cdot|s_t)\,\|\,\pi_{new}(\cdot|s_t))\right] \le \delta
$$

  - A2C actor objective:

$$
L^{A2C}_{actor} = -\mathbb{E}_t\left[\log\pi_\theta(a_t|s_t)\,\hat{A}_t\right]
$$

  - SAC actor objective:

$$
J_\pi = \mathbb{E}_{s_t\sim\mathcal{D},a_t\sim\pi}\left[\alpha\log\pi(a_t|s_t) - \min_i Q_i(s_t,a_t)\right]
$$

### Compare on
- **What it does:** Enables compare mode for parallel multi-run training.
- **How implemented:** In `_start_worker(...)`, if compare is on and not single-episode, `_start_compare_workers(...)` is used.
- **Policy dependency:** Compare may include policies and non-policy parameters; mode constraints still apply.

### Clear (button)
- **What it does:** Clears all stored compare parameter lists.
- **How implemented:** `_clear_compare_parameter_lists()` clears `_compare_param_lists`, `_compare_raw_lists`, input field, and summary text.
- **Policy dependency:** None directly.

### Parameter (dropdown)
- **What it does:** Selects which parameter you want to vary in compare mode.
- **How implemented:** Driven by `_compare_parameter_specs()` mapping names to snapshot keys and data types.
- **Policy dependency:** Includes policy and policy-specific hyperparameters; valid values are type-checked.

### Options (text input)
- **What it does:** Comma-separated list of values for selected compare parameter.
- **How implemented:** `_parse_compare_values()` parses and validates by type (`int`, `float`, `bool`, `option`, `str`), removes duplicates.
- **Policy dependency:**
  - For enum fields (e.g., `Policy`, `Activation`, `LR strategy`), values must be valid options.
  - Policy list can include both discrete and continuous policies; each generated run auto-forces compatible environment mode.

### Compare summary (text output)
- **What it does:** Shows active compare lists as `Parameter: [v1, v2]`.
- **How implemented:** `_refresh_compare_summary()` from `_compare_param_lists`.
- **Policy dependency:** Reflects any policy-specific lists user entered.

### Compare combination behavior
- **What it does:** Runs all Cartesian combinations of entered compare lists.
- **How implemented:** `_build_compare_run_configs()` uses `itertools.product(...)`, merges each combo into snapshot, then starts one worker per combo.
- **Policy dependency:**
  - For each combo, policy determines discrete vs continuous mode and agent class.
  - Any parameter not included in compare lists uses the normal GUI value.
- **Formula:** if you define $k$ compare parameters with list sizes $n_1, n_2, \dots, n_k$, total runs are

$$
N_{runs} = \prod_{i=1}^{k} n_i
$$

---

## 3) General Group

### Policy
- **What it does:** Chooses RL algorithm.
- **How implemented:** `_on_policy_changed()` applies defaults (`_apply_policy_defaults`) and enforces env mode.
- **Policy dependency:** Core selector.
  - **Discrete:** `DuelingDQN`, `D3QN`, `DDQN+PER`
  - **Continuous:** `PPO`, `A2C`, `TRPO`, `SAC`

### Max steps
- **What it does:** Max environment steps per episode.
- **How implemented:** Passed to `Trainer.run_episode(..., max_steps=...)` and training loops.
- **Policy dependency:** All policies; also used in training plan for discrete policies.

### Episodes
- **What it does:** Number of episodes in Train and Run.
- **How implemented:** Passed to `Trainer.train(...)` and `Trainer.set_training_plan(...)`.
- **Policy dependency:**
  - Discrete: contributes to planned scheduler horizon via total steps/cadence.
  - Continuous: contributes via policy-specific updates-per-episode mapping.

### Epsilon max
- **What it does:** Initial epsilon.
- **How implemented:** `epsilon` initialized in worker loop.
- **Policy dependency:**
  - **Discrete policies:** used directly for epsilon-greedy action selection.
  - **Continuous policies:** currently effectively ignored by action selection (policy sampling is used).

### Epsilon decay
- **What it does:** Multiplicative decay per episode.
- **How implemented:** `epsilon = max(epsilon_min, epsilon * epsilon_decay)` after each episode.
- **Policy dependency:** Same as epsilon max (effective for discrete, largely inert for continuous).
- **Formula:**

$$
\epsilon_{t+1} = \max(\epsilon_{min},\; \epsilon_t \cdot d)
$$

with decay factor $d=\text{epsilon\_decay}$. Before clamping:

$$
\epsilon_t = \epsilon_0 d^t
$$

### Epsilon min
- **What it does:** Lower bound for epsilon.
- **How implemented:** Clamp in episode loop.
- **Policy dependency:** Same as epsilon max.

---

## 4) Specific Group (Policy Hyperparameters)

### Gamma
- **What it does:** Discount factor.
- **How implemented:**
  - Discrete: target computation `r + (1-done) * gamma * Q_target(...)`.
  - Continuous (`PPO/A2C/TRPO`): advantage/return accumulation backward through trajectory.
  - Continuous (`SAC`): Bellman target over twin target critics with entropy term.
- **Policy dependency:** All policies, but applied through different update formulas.
- **Formulas:**

Discrete TD target:

$$
y_t = r_t + (1-d_t)\,\gamma\,Q_{target}(s_{t+1}, a^*)
$$

Trajectory return used in continuous path:

$$
G_t = r_t + \gamma G_{t+1}
$$

SAC soft-Q target:

$$
y_t = r_t + (1-d_t)\,\gamma\left(\min_i Q^{target}_i(s_{t+1}, a_{t+1}) - \alpha\log\pi(a_{t+1}|s_{t+1})\right)
$$

### Learning rate
- **What it does:** Base optimizer step size.
- **How implemented:**
  - Discrete: Adam on Q-network.
  - Continuous: Adam on actor/log_std and critic.
  - Can be modified over time by LR scheduler strategy.
- **Policy dependency:** All policies; sensitivity differs (continuous policies often more LR-sensitive).
- **Formula (generic gradient step):**

$$
	\theta_{t+1} = \theta_t - \eta_t\,\nabla_\theta \mathcal{L}_t
$$

where $\eta_t$ is the scheduled learning rate.

### Replay size
- **What it does:** Replay buffer capacity.
- **How implemented:**
  - Discrete: sets buffer max length.
  - Continuous (`SAC`): sets off-policy replay max length.
  - Continuous (`PPO/A2C/TRPO`): not used (on-policy trajectory buffer).
- **Policy dependency:** Discrete and `SAC` directly; other continuous policies none.

### Batch size
- **What it does:** Minibatch size for replay updates.
- **How implemented:**
  - Discrete: used in replay sampling and `can_learn()` gate.
  - Continuous (`SAC`): used for replay sampling in each gradient step.
  - Continuous (`PPO/A2C/TRPO`): used for rollout minibatch slicing in policy/value updates.
- **Policy dependency:** All policies.
- **Formula (sampled batch):** for $B=\text{batch\_size}$,

$$
\mathcal{B} = \{(s_i,a_i,r_i,s'_i,d_i)\}_{i=1}^{B}
$$

### Target update
- **What it does:** Frequency of copying online network to target network.
- **How implemented:**
  - Discrete: hard sync when `learn_steps % target_update == 0`.
  - Continuous (`SAC`): soft target-critic updates (Polyak style) during learning.
  - Continuous (`PPO/A2C/TRPO`): not used.
- **Policy dependency:** Discrete and `SAC`.
- **Formula:** if step $t$ satisfies $t \bmod K = 0$ with $K=\text{target\_update}$,

$$
	\theta^- \leftarrow \theta
$$

### Replay warmup
- **What it does:** Minimum replay size before learning starts.
- **How implemented:**
  - Discrete: `can_learn()` checks `len(replay) >= replay_warmup`.
  - Continuous (`SAC`): same warmup gate on continuous replay.
  - Continuous (`PPO/A2C/TRPO`): not used.
- **Policy dependency:** Discrete and `SAC`.
- **Condition:** learning enabled only if

$$
|\mathcal{D}| \ge W
$$

where $|\mathcal{D}|$ is replay size and $W=\text{replay\_warmup}$.

### Learning cadence
- **What it does:** Learn every N environment steps.
- **How implemented:**
  - Discrete: `can_learn()` checks `total_steps % cadence == 0`.
  - Continuous (`SAC`): same step-gated cadence after warmup.
  - Continuous (`PPO/A2C/TRPO`): rollout update cadence (agent updates every `learning_cadence` collected steps, plus one end-of-episode flush for leftover trajectory).
- **Policy dependency:**
  - Discrete: replay update cadence.
  - Continuous (`SAC`): replay update cadence.
  - Continuous (`PPO/A2C/TRPO`): rollout update cadence.
- **Condition:** perform update when

$$
t \bmod C = 0
$$

with cadence $C=\text{learning\_cadence}$.

### Activation
- **What it does:** Nonlinearity in neural networks.
- **How implemented:** `make_activation(...)` used while constructing MLP layers.
- **Policy dependency:** All policies (Q nets and actor/critic networks).

### Hidden layers
- **What it does:** Network architecture as comma-separated sizes (e.g. `256,256`).
- **How implemented:** `parse_hidden_layers(...)` -> MLP/Q network construction.
- **Policy dependency:** All policies.

### LR strategy
- **What it does:** Chooses LR schedule behavior.
- **How implemented:** `_step_lr_schedule(...)` in both discrete and continuous agents.
- **Options behavior:**
  - `exponential`: multiplicative interpolation toward target LR.
  - `linear`: linear interpolation toward target LR.
  - `cosine`: cosine decay toward target LR.
  - `loss-based`: decays when loss stagnates (`_loss_patience`, `_loss_tolerance`).
  - `guarded natural gradient`: scheduled LR scaled by gradient norm guard.
- **Policy dependency:** All policies.
- **Formulas:** let $\eta_0$ be base LR, $\eta_{min}$ min LR, and $\eta_T=\max(\eta_{min},\eta_0\cdot\text{lr\_decay})$.

Progress:

$$
p = \min\left(1,\frac{\text{learn\_steps}}{\text{planned\_steps}}\right)
$$

Linear:

$$
\eta(p)=\eta_0 + (\eta_T-\eta_0)p
$$

Cosine:

$$
\eta(p)=\eta_T + \tfrac{1}{2}(\eta_0-\eta_T)(1+\cos(\pi p))
$$

Exponential interpolation:

$$
\eta(p)=\eta_0\left(\frac{\eta_T}{\eta_0}\right)^p
$$

Guarded natural-gradient variant (implemented guard):

$$
\eta = \frac{\eta_{scheduled}}{1 + g\,\|\nabla\|}
$$

with $g=\text{lr\_decay}$.

### LR decay
- **What it does:** Target-LR ratio / decay strength depending on strategy.
- **How implemented:**
  - Defines `_target_lr = base_lr * lr_decay` (clamped by min LR).
  - Used as factor in `loss-based` and guard strength in `guarded natural gradient`.
- **Policy dependency:** All policies.

### Min LR
- **What it does:** Lower floor for learning rate.
- **How implemented:** `_set_optimizer_lr(...)` clamps LR to `>= min_learning_rate`.
- **Policy dependency:** All policies.
- **Formula:**

$$
\eta \leftarrow \max(\eta, \eta_{min})
$$

### GAE λ
- **What it does:** Controls bias-variance tradeoff in generalized advantage estimation for continuous-policy updates.
- **How implemented:** Used in `BaseContinuousAgent._update_from_trajectory()` when computing GAE from TD deltas.
- **Policy dependency:** Used by continuous policies (`PPO`, `A2C`, `TRPO`, `SAC`) in current implementation.
- **Formula:**

$$
\hat A_t^{GAE(\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},\quad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### PPO clip
- **What it does:** Limits PPO policy-ratio update size around 1.0.
- **How implemented:** Used in PPO actor loss with ratio clipping in continuous update loop.
- **Policy dependency:** Directly affects `PPO`; ignored by non-PPO policies.
- **Formula:**

$$
L^{PPO}_{clip} = -\min\left(r_t\hat A_t,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)
$$

with $\epsilon=\text{ppo\_clip\_range}$.

---

## 5) Live Plot Group

### Moving average values
- **What it does:** Window length for moving-average reward line.
- **How implemented:** `_moving_average(values, window)` for preview/finalized runs.
- **Policy dependency:** None. Visualization only.
- **Formula:** for rewards $r_1,\dots,r_t$ and window $W$,

$$
	ext{MA}_t = \frac{1}{m}\sum_{i=t-m+1}^{t} r_i,\quad m=\min(W,t)
$$

---

## 6) Policy-Specific Notes (Important)

## Discrete policies (`DuelingDQN`, `D3QN`, `DDQN+PER`)
- Use epsilon-greedy exploration (`epsilon max/decay/min` are effective).
- Use replay buffer parameters (`replay size`, `batch size`, `replay warmup`, `learning cadence`).
- Use target network (`target update`).
- `DDQN+PER` additionally uses prioritized replay sampling.

## Continuous policies (`PPO`, `A2C`, `TRPO`, `SAC`)
- Use stochastic policy sampling from Normal actor distribution + tanh squashing.
- Epsilon parameters are currently passed through worker loops but not used by action selection.
- `learning cadence` is used as rollout update cadence in the on-policy path.
- Replay-related settings (`replay size`, `batch size`, `replay warmup`) and `target update` are currently not part of the on-policy update path.
- End-of-episode updates differ by algorithm (`epochs` and objective terms differ).
- Continuous training uses additional stabilization internals (entropy coefficients, gradient clipping, SmoothL1 critic loss, NaN guards).
- **Core formulas used in implementation:**
  - Advantages (normalized):

$$
A_t = G_t - V_\phi(s_t),\quad \hat{A}_t = \frac{A_t-\mu_A}{\sigma_A+\epsilon}
$$

  - PPO clipped objective term:

$$
r_t(\theta)=\exp(\log\pi_\theta - \log\pi_{old}),\quad
L^{PPO}= -\min\big(r_t\hat{A}_t,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t\big)
$$

  - Critic loss (Smooth L1 / Huber):

$$
L_V = \text{SmoothL1}(V_\phi(s_t), G_t)
$$

  - Combined continuous loss form in code:

$$
L = L_{actor} + 0.5\,L_V - \beta\,\mathcal{H}
$$

with policy-dependent $L_{actor}$ and entropy coefficient $\beta$.

---

## 7) Compare Mode + Policy Interaction

- Compare mode creates one run per parameter combination.
- Each run captures immutable metadata used for labels and finalization.
- If `Policy` is part of compare lists, each run enforces compatible `continous` mode automatically.
- In mixed compare sets, non-compared fields remain fixed from the base GUI values.

---

## 8) Practical Tuning Guidance

- Start with one-parameter sweeps (e.g. learning rate) before multi-parameter combinations.
- For discrete policies, tune replay warmup/cadence together with LR.
- For continuous policies, prioritize LR + LR strategy/decay + gamma first.
- Keep compare list sizes moderate; Cartesian products grow quickly.
