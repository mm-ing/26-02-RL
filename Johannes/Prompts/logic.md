# Logic Prompt Blueprint

## Program Architecture and Logic

### Logic Module
Implement:
- environment wrapper class
- SB3 policy agent wrapper
- trainer class

Use this separation:
- `Algorithm` layer: model construction, action selection, update primitives
- `TrainLoop` layer: episode/step orchestration, stop/pause handling, callback/event emission
- keep UI-independent orchestration reusable for headless runs/tests

Trainer must provide:
- `run_episode(...)`
- `train(...)`
- `evaluate_policy(...)`
- environment rebuild/update
- optional CSV export of sampled transitions
- periodic deterministic evaluation checkpoints

Event contract baseline (logic-to-GUI bridge):
- episode updates emitted as `type = "episode"`
- completion emitted as `type = "training_done"` (snake_case)
- failures emitted as `type = "error"`
- include run/session identifier tags in worker-originated events so the GUI can ignore stale events from canceled or replaced runs
- attach `session_id` in the worker event sink/bridge layer (GUI-side worker wrapper), not by injecting unknown keys into strict trainer config constructors
- episode payload should include `run_id`, `episode`, `episodes`, `reward`, `moving_average`, `eval_points`, `steps`, `epsilon`, `lr`, `best_reward`, and `render_state`
- for animation updates, episode payload should also include rollout frames (preferred key: `frames`; optional compatibility key: `frame`)

---

## SB3 Policy Behavior
- expose exactly the policy list from the project-specific file
- keep GUI display names even if internal mapping needs valid identifiers
- keep per-policy default snapshots
- maintain per-policy runtime parameter snapshots for all `Specific` group parameters (shared + policy-specific) so switching policies restores that policy’s own values/defaults
- reinitialize weights for each new `Train and Run`
- separate deterministic evaluation from exploration episodes
- tune per-policy defaults for the target environment
- for exposed NN-based policies, support shared NN parameter controls (`hidden_layer`, `lr_strategy`, `min_lr`, `lr_decay`) and apply them in model construction
- hidden-layer controls should affect policy network architecture (for example `policy_kwargs.net_arch` for SB3 MLP policies)
- hidden-layer parsing rule:
	- single value (for example `256`) maps to symmetric two-layer `[256, 256]`
	- comma-separated values (for example `256,128,64`) map directly to multi-layer architecture `[256, 128, 64]`
	- invalid/empty values should safely fall back to defaults
- learning-rate strategy controls should drive schedule behavior (`constant`, `linear`, `exponential`) with a `min_lr` floor and `lr_decay` factor for exponential mode
- for `PPO`, keep settings consistent (`n_steps >= batch_size`, divisibility preferred)
- for `SAC`/`TD3`, use realistic replay warmup and buffer sizes
- support runtime device selection (`CPU` / `GPU`) with default `CPU`
- only include algorithm-specific parameters that are explicitly wired into the selected SB3 model constructor; avoid dead/unconsumed UI/config fields

Typical mapping (if needed):
- DQN-like labels -> `stable_baselines3.DQN`
- PPO -> `stable_baselines3.PPO`
- A2C -> `stable_baselines3.A2C`
- SAC -> `stable_baselines3.SAC`
- TRPO -> `sb3_contrib.TRPO` (only if explicitly required)

---

## Runtime Behavior and Responsiveness
- training runs in background thread(s)
- worker threads write pending state only (thread-safe)
- periodic UI pump consumes pending updates
- before a new run, flush queued worker events so finalize data is not lost
- no worker-side GUI drawing
- support bounded compare parallelism (max concurrent workers: `4`)
- support latest-frame-wins rendering (don’t enqueue all frames)
- optional fast non-render evaluation path for non-visual episodes/checkpoints
- default/runtime device should be `CPU` with selectable `CPU` / `GPU`
- if `GPU` is selected but CUDA is unavailable, safely fall back to `CPU`
- pause/resume/cancel controls must operate on active worker trainers, including compare workers (not on an unused template trainer instance)
- use pendulum-style pause gating for worker loops: `pause_event.set()` means running, `pause_event.clear()` means paused, and loops should block with `pause_event.wait()`

### Update-Rate Behavior
Use `Update rate (episodes)` as gating interval:
- reward data stored every episode
- animation updates on every Nth episode (and final episode)
- live plot refresh every episode
- replay frame capture uses fixed stride `Frame stride = N` (default `2`): capture first step and then every Nth step
- when animation is enabled, rollout environment must be created with render mode supporting frame extraction (`rgb_array`)
- during training animation, capture and emit a sampled frame sequence for replay (not a single final frame only)
- replay queue behavior: allow at most one pending playback while one is active; if newer frames arrive, replace pending playback with newest (`latest-wins`) and keep current playback uninterrupted
- deterministic evaluation rollouts/plot points should run on a fixed cadence of every 10th episode

---

## Compare Mode Behavior
- treat each Cartesian combination as one run
- execute combinations with max `4` concurrent workers
- when running compare workers, use adaptive per-worker CPU thread budgeting derived from available CPU cores and active worker count to balance throughput vs. oversubscription
- do not force user-selected `GPU` device to `CPU` solely because compare mode is enabled; honor selected device with normal availability fallback rules
- assign a unique internal run ID per compare combination so plot/history state cannot be overwritten by timestamp collisions
- avoid duplicate finalize lines
- keep per-run label metadata immutable
- let the selected policy drive render/status/progress when available
- if `Policy` is compared, start from policy defaults; only explicit compared values override
- when `Policy` is compared, apply compared values only when parameter keys are valid for that selected policy; ignore policy-incompatible compare keys
- in compare mode, animation should be enabled for exactly one selected render run (selected policy preferred; fallback first run) and disabled for other workers to avoid mixed/flickering playback
- register worker trainer handles before worker submission so pause/cancel can act immediately after training starts

---

## Exports
- CSV export to `results_csv/`
- PNG export to `plots/`
- PNG filename includes run params + timestamp

---

## Logic Tests
- logic tests: environment/trainer/run_episode/evaluate core behavior
- include headless smoke path for training loop + event propagation
- verify pause/resume/cancel transitions and final status reporting
- verify `run_episode` returns actual executed environment steps even if transition collection is disabled
- verify deterministic evaluation checkpoint cadence (for example every Nth episode) populates eval points at expected episodes
- verify sampled transition CSV export contains expected columns and non-empty data after transition collection
- verify compare-policy baseline behavior: each compared policy starts from its own defaults before explicit compare overrides
- verify compare-policy override applicability: policy-incompatible compare parameters are ignored safely
- verify compare worker CPU thread budgeting distributes core budget as expected
- verify policy snapshot isolation: shared specific parameters (`gamma`, `learning_rate`, `batch_size`, `hidden_layer`, `lr_strategy`, `min_lr`, `lr_decay`) and policy-specific parameters remain independent across policy switches
- verify NN parameter wiring: hidden-layer and LR schedule controls are consumed by logic/model construction for all exposed NN policies

Use this test isolation setup (required in multi-project workspaces):
- local `pytest.ini` with `testpaths = tests`
- isolated run command: `python -m pytest -q --rootdir . --confcutdir . tests/...`
- optional local `run_tests.py` helper
