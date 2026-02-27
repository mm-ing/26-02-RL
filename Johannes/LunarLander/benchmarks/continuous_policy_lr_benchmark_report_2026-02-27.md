# Continuous Policy LR Benchmark Report (2026-02-27)

## Scope
- Environment: `LunarLander-v3` with `continuous=True`
- Policies: `PPO`, `A2C`, `TRPO`, `SAC`
- Seeds: `0, 1, 2`
- Episodes: `25`
- Max steps: `300`
- Metric: `median tail-8 reward` (mean of last 8 episode rewards per seed, then median across seeds)

## Results

| Policy | Candidate | LR | Strategy | Decay | Median tail-8 | Mean tail-8 | Seed scores |
|---|---|---:|---|---:|---:|---:|---|
| PPO | default | 7.50e-05 | linear | 0.30 | -256.043 | -261.460 | [-256.043, -252.929, -275.408] |
| PPO | alt | 1.00e-04 | linear | 0.30 | **-229.679** | -204.784 | [-229.679, -265.492, -119.182] |
| A2C | default | 1.50e-04 | exponential | 0.30 | **-183.283** | -175.336 | [-183.283, -201.439, -141.287] |
| A2C | alt | 1.00e-04 | exponential | 0.30 | -225.470 | -223.358 | [-225.470, -208.779, -235.824] |
| TRPO | default | 7.50e-05 | linear | 0.40 | **-162.719** | -182.700 | [-245.017, -162.719, -140.365] |
| TRPO | alt | 1.00e-04 | linear | 0.40 | -241.713 | -248.298 | [-241.713, -316.912, -186.269] |
| SAC | default | 1.00e-04 | cosine | 0.30 | **-129.201** | -145.197 | [-208.654, -97.737, -129.201] |
| SAC | alt | 7.50e-05 | cosine | 0.30 | -183.226 | -175.810 | [-183.226, -243.613, -100.592] |

## Recommended Defaults
- `PPO`: `1.00e-04`, `linear`, `0.30`
- `A2C`: `1.50e-04`, `exponential`, `0.30`
- `TRPO`: `7.50e-05`, `linear`, `0.40`
- `SAC`: `1.00e-04`, `cosine`, `0.30`

## Reproducibility
Run:

```bash
python benchmarks/continuous_policy_lr_benchmark.py
```

The script is located at `benchmarks/continuous_policy_lr_benchmark.py`.
