# Policies.md

This document explains all RL policies used in the LunarLander project:
- theory (with core formulas),
- implementation in this codebase,
- relevant parameters,
- practical effect of those parameters.

The project contains 7 policies:
- Discrete: `DuelingDQN`, `D3QN`, `DDQN+PER`
- Continuous: `PPO`, `A2C`, `TRPO`, `SAC`

## One-page cheat sheet

| Policy | Default core settings | First 3 tuning knobs |
|---|---|---|
| DuelingDQN | `lr=3e-4`, `cadence=2`, `replay_warmup=5000`, `batch=128` | `learning_rate`, `target_update`, `replay_warmup` |
| D3QN | `lr=2.5e-4`, `cadence=2`, `replay_warmup=8000`, `batch=128` | `learning_rate`, `target_update`, `batch_size` |
| DDQN+PER | `lr=2e-4`, `cadence=2`, `replay_warmup=10000`, `batch=128` | `learning_rate`, `replay_size`, `replay_warmup` |
| PPO | `lr=1e-4`, `cadence=32`, `gae_lambda=0.95`, `clip=0.2` | `learning_rate`, `ppo_clip_range`, `learning_cadence` |
| A2C | `lr=1.5e-4`, `cadence=16`, `gae_lambda=1.0`, `batch=128` | `learning_rate`, `gae_lambda`, `learning_cadence` |
| TRPO | `lr=7.5e-5`, `cadence=32`, `gae_lambda=0.95`, `batch=128` | `gae_lambda`, `learning_cadence`, `batch_size` (critic) |
| SAC | `lr=1e-4`, `cadence=32`, `replay_warmup=10000`, `batch=128` | `learning_rate`, `replay_warmup`, `batch_size` |

**Rule of thumb**
- If unstable: lower `learning_rate` first.
- If too noisy: increase `learning_cadence` (or `batch_size`).
- If too slow: reduce `replay_warmup` (off-policy) or cadence carefully.

## Quick comparison table

| Policy | Family | Action space | Core idea | Strengths | Most sensitive parameters | Default use-case |
|---|---|---|---|---|---|---|
| DuelingDQN | Off-policy value-based | Discrete | Dueling value+advantage Q-heads with Double-Q target | Solid discrete baseline, stable learning | `learning_rate`, `target_update`, `replay_warmup`, `learning_cadence` | First stable discrete baseline |
| D3QN | Off-policy value-based | Discrete | Double + Dueling architecture | Better bias/stability tradeoff than plain DQN variants | `learning_rate`, `target_update`, `batch_size`, `replay_size` | Higher-capacity discrete training |
| DDQN+PER | Off-policy value-based | Discrete | Double-Q + prioritized replay | High sample efficiency on informative transitions | `learning_rate`, `replay_size`, `batch_size`, `replay_warmup` | Discrete runs with limited sample budget |
| PPO | On-policy actor-critic | Continuous | Clipped policy-ratio objective + GAE | Strong robust continuous baseline | `learning_rate`, `ppo_clip_range`, `gae_lambda`, `learning_cadence` | Default continuous baseline |
| A2C | On-policy actor-critic | Continuous | Advantage-weighted policy gradient + value loss | Simpler/faster iteration loop | `learning_rate`, `gae_lambda`, `batch_size`, `learning_cadence` | Fast continuous prototyping |
| TRPO | On-policy actor-critic | Continuous | KL-constrained trust-region policy step | Conservative stable policy updates | `gae_lambda`, `learning_cadence`, `batch_size` (critic), trust-region internals | Stability-critical continuous updates |
| SAC | Off-policy actor-critic | Continuous | Entropy-regularized objective with twin Q critics | Strong long-run performance, sample efficient | `learning_rate`, `replay_warmup`, `batch_size`, `learning_cadence`, entropy temperature | Long-run continuous performance focus |

---

## 1) Shared setup and notation

### Environment/action-space split
- **Discrete policies** operate with `continuous=False` and integer actions.
- **Continuous policies** operate with `continuous=True` and vector actions in $[-1,1]$ after `tanh` squashing.

### Common symbols
- $s_t, a_t, r_t, s_{t+1}, d_t$: state, action, reward, next state, done flag.
- $\gamma$: discount factor (`gamma`).
- $\eta$: learning rate (`learning_rate`, then scheduler-modified).
- $\mathcal{D}$: replay buffer.
- $A_t$: advantage estimate.

### LR scheduler block (used across policies)
Configured via:
- `lr_strategy` in `{exponential, linear, cosine, loss-based, guarded natural gradient}`
- `lr_decay`
- `min_learning_rate`

The scheduler changes the optimizer LR over planned training steps and can react to poor loss progress (`loss-based`) or large gradients (`guarded natural gradient`).

---

## 2) Discrete family

## 2.1 DuelingDQN

### Theory
Dueling architecture decomposes Q-values into value and advantage:

$$
Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')\right)
$$

This helps learning state-value quality even when action advantages are similar.

The project uses Double-DQN targets:

$$
y_t = r_t + (1-d_t)\,\gamma\,Q_{target}(s_{t+1}, \arg\max_a Q_{online}(s_{t+1},a))
$$

### Implementation in this project
- Class: `DuelingDQN(BaseDQNAgent)`.
- Network: dueling heads in `QNetwork` (`value_head`, `advantage_head`).
- Learning:
  - sample minibatch from uniform replay,
  - compute SmoothL1 (Huber) TD loss,
  - gradient clipping,
  - hard target copy every `target_update` learn steps.

### Main parameters and effects
- `gamma`: larger => more long-term planning, potentially slower stabilization.
- `learning_rate`: too high destabilizes TD learning; too low slows convergence.
- `replay_size`: bigger memory diversity, but slower adaptation to very recent dynamics.
- `batch_size`: larger reduces gradient noise but increases compute cost.
- `target_update`: smaller value updates target net more often (faster but potentially less stable).
- `replay_warmup`: prevents early overfitting on tiny replay.
- `learning_cadence`: update frequency; lower cadence value => more frequent updates.
- `hidden_layers`, `activation_function`: model capacity and nonlinearity.

---

## 2.2 D3QN (Double + Dueling)

### Theory
D3QN combines:
- Double Q-learning target selection/evaluation split,
- Dueling value/advantage decomposition.

Compared with plain DQN, it reduces overestimation bias and often improves sample efficiency.

### Implementation in this project
- Class: `D3QN(BaseDQNAgent)`.
- Internally same training core as `DuelingDQN` (`dueling=True`, uniform replay), with its own tuned defaults.

### Main parameters and effects
Same set as DuelingDQN; in practice:
- slightly lower LR and larger network are often used to exploit richer representation,
- replay and target cadence strongly affect stability.

---

## 2.3 DDQN+PER

### Theory
DDQN reduces overestimation; PER (Prioritized Experience Replay) samples transitions by priority:

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}},\quad p_i \approx |\delta_i| + \epsilon
$$

Importance-sampling correction (implemented as weights in loss) compensates sampling bias:

$$
L = \frac{1}{B}\sum_i w_i\,\ell_i
$$

### Implementation in this project
- Class: `DDQNPER(BaseDQNAgent)` with `prioritized=True`.
- Uses `PrioritizedReplayBuffer`:
  - samples by priority distribution,
  - returns IS weights,
  - updates priorities from TD errors after each step.

### Main parameters and effects
- All DQN parameters above plus PER-internal behavior:
  - TD-error scale indirectly controls future sampling probability.
- Practical effect:
  - faster focus on informative transitions,
  - can over-focus noisy outliers if LR too high or replay too small.

---

## 3) Continuous family

## 3.1 PPO (true continuous on-policy path)

### Theory
PPO uses clipped policy-ratio objective:

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)},\quad
L^{clip}=\mathbb{E}\left[\min\left(r_t A_t,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]
$$

Value learning is combined with actor objective.

Advantages are built with GAE:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\quad
A_t^{GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}
$$

### Implementation in this project
- Class: `PPOAgent(BaseContinuousAgent)`.
- Action distribution: Normal policy with `tanh` squashing.
- Update path:
  - collect rollout trajectory,
  - compute normalized GAE advantages,
  - multiple epochs/minibatches,
  - clipped PPO surrogate + critic SmoothL1 + entropy regularization.

### Main parameters and effects
- `ppo_clip_range`: bigger clip => larger policy moves, faster but riskier.
- `gae_lambda`: lower values reduce variance but add bias.
- `learning_cadence`: rollout length before update; too short increases noise.
- `batch_size`: minibatch size inside rollout updates.
- `gamma`, `learning_rate`, LR scheduler trio: core stability/speed controls.

---

## 3.2 A2C (dedicated true actor-critic path)

### Theory
A2C policy gradient with baseline:

$$
L_{actor}^{A2C} = -\mathbb{E}[\log\pi_\theta(a_t|s_t)\,A_t]
$$

Critic learns value target (here advantage-derived returns):

$$
L_V = \text{SmoothL1}(V_\phi(s_t), R_t)
$$

### Implementation in this project
- Class: `A2CAgent(BaseContinuousAgent)` with **its own** `_update_from_trajectory`.
- Differences vs PPO path:
  - no ratio clipping,
  - no trust-region step,
  - single-pass minibatch actor-critic update per collected rollout.

### Main parameters and effects
- `gae_lambda`: central bias/variance dial in advantage signal.
- `learning_cadence`: controls rollout size and therefore gradient variance.
- `batch_size`: controls minibatch granularity over rollout.
- `learning_rate`, `gamma`, `entropy` effect (indirect via implementation constant): exploration vs convergence speed.

---

## 3.3 TRPO (dedicated trust-region path)

### Theory
TRPO maximizes surrogate objective under KL constraint:

$$
\max_\theta\; \mathbb{E}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}A_t\right]
\quad \text{s.t.} \quad
\mathbb{E}[D_{KL}(\pi_{old}(\cdot|s_t)\|\pi_\theta(\cdot|s_t))] \le \delta
$$

Natural-gradient style step is approximated with conjugate gradient and line search.

### Implementation in this project
- Class: `TRPOAgent(BaseContinuousAgent)` with dedicated `_update_from_trajectory`.
- Policy update sequence:
  1. compute surrogate gradient,
  2. compute Fisher-vector products,
  3. solve step direction via conjugate gradient,
  4. scale to KL budget (`_max_kl`),
  5. backtracking line search over candidate steps,
  6. accept only if improvement > 0 and KL bound satisfied.
- Critic is fitted separately by minibatches.

### Main parameters and effects
- `gae_lambda`, `gamma`: quality/stability of advantage estimate.
- `learning_rate`: affects critic optimizer and scheduler state.
- `batch_size`: affects critic fitting batches (policy step remains full-batch).
- `learning_cadence`: rollout size before each trust-region update.
- Internal TRPO controls (in code):
  - `_max_kl` (trust-region size),
  - `_cg_iters`, `_cg_damping`, line-search fractions.

---

## 3.4 SAC (dedicated true off-policy path)

### Theory
SAC optimizes entropy-regularized objective:

$$
J_\pi = \mathbb{E}[\alpha\log\pi(a|s) - Q(s,a)]
$$

Twin critics reduce overestimation:

$$
Q_{target}=r + (1-d)\gamma\left(\min(Q_1',Q_2') - \alpha\log\pi(a'|s')\right)
$$

Temperature $\alpha$ is learned (here via `log_alpha`) to match target entropy.

### Implementation in this project
- Class: `SACAgent` (standalone, not using BaseContinuous rollout updater).
- Components:
  - actor mean net + actor log-std net,
  - twin critics (`q1`, `q2`) and twin targets,
  - continuous replay buffer,
  - reparameterized action sampling (`rsample`) with tanh log-prob correction,
  - alpha optimizer on `log_alpha`.
- Update sequence per learn step:
  1. sample replay minibatch,
  2. compute target with target critics and entropy term,
  3. optimize `q1`, `q2`,
  4. optimize actor,
  5. optimize `alpha`,
  6. soft-update target critics (Polyak).

### Main parameters and effects
- `replay_size`, `replay_warmup`: essential for off-policy stability.
- `batch_size`: core tradeoff between stable gradients and throughput.
- `learning_cadence`: frequency gate for SAC gradient steps.
- `gamma`, `learning_rate`, LR schedule: convergence speed/stability.
- `target_update` is conceptually represented as continuous soft updates in this implementation.

---

## 4) Policy-specific parameter relevance matrix

### Mostly discrete-relevant
- `epsilon max`, `epsilon decay`, `epsilon min`
  - Strong effect for DQN-family action selection.
  - Minimal/no direct effect on continuous policy action sampling.

### Shared across all policies
- `gamma`, `learning_rate`, `lr_strategy`, `lr_decay`, `min_learning_rate`,
- `hidden_layers`, `activation_function`,
- `learning_cadence` (semantics differ),
- `batch_size` (usage differs).

### Mostly off-policy replay-centric
- `replay_size`, `replay_warmup`, `target_update`
  - Core for DQN-family and SAC.
  - Not used as core mechanism in PPO/A2C/TRPO policy updates.

### Continuous on-policy specific
- `gae_lambda`, `ppo_clip_range`
  - `gae_lambda`: used in PPO/A2C/TRPO advantage estimation.
  - `ppo_clip_range`: used by PPO clipped objective.

---

## 5) Practical “which policy when?”

- `DuelingDQN` / `D3QN`: robust default for discrete mode, easier to debug.
- `DDQN+PER`: often best sample efficiency in discrete mode, with more replay sensitivity.
- `PPO`: strong general baseline for continuous mode.
- `A2C`: simpler continuous baseline; often faster iterations, may be noisier.
- `TRPO`: conservative/stable policy updates when KL control is desired.
- `SAC`: strong off-policy continuous learner, usually good long-run performance.

## 5.1 Recommended starting order

- **Discrete track:** `DuelingDQN` -> `D3QN` -> `DDQN+PER`
  - Start with `DuelingDQN` for baseline stability,
  - move to `D3QN` for stronger bias/stability tradeoff,
  - finish with `DDQN+PER` when you want better sample efficiency.

- **Continuous track:** `PPO` -> `SAC` -> `TRPO` (use `A2C` for fast prototyping)
  - Start with `PPO` as robust baseline,
  - move to `SAC` for stronger long-run performance,
  - use `TRPO` when strict update stability is more important than speed,
  - use `A2C` early for quick iteration and debugging.

---

## 6) Common failure modes and tuning hints

- **Unstable/NaN losses**:
  - lower `learning_rate`,
  - increase `learning_cadence` (larger rollouts or less frequent updates),
  - verify `min_learning_rate` not too high.

- **Very slow learning**:
  - increase LR moderately,
  - reduce `learning_cadence` for more frequent updates,
  - check action mode (`continuous`/`discrete`) matches policy family.

- **High variance across runs**:
  - compare by median over multiple seeds,
  - use slightly larger `batch_size`,
  - for PPO/TRPO/A2C tune `gae_lambda` around `0.9-0.97`.

- **Low throughput with GPU in short runs**:
  - this project often runs faster on CPU for small-step RL loops,
  - GPU may still give better reward consistency in longer runs.

---

## 7) Current project defaults (for quick reference)

- `DuelingDQN`: lr `3e-4`, cadence `2`, replay warmup `5000`
- `D3QN`: lr `2.5e-4`, cadence `2`, replay warmup `8000`
- `DDQN+PER`: lr `2e-4`, cadence `2`, replay warmup `10000`
- `PPO`: lr `1e-4`, cadence `32`, `gae_lambda=0.95`, `ppo_clip_range=0.2`
- `A2C`: lr `1.5e-4`, cadence `16`, `gae_lambda=1.0`
- `TRPO`: lr `7.5e-5`, cadence `32`, `gae_lambda=0.95`
- `SAC`: lr `1e-4`, cadence `32`, replay warmup `10000`

(Full defaults are defined in `POLICY_DEFAULTS` in `LunarLander_logic.py`.)
