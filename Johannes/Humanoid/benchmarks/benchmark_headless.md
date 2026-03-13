# Humanoid Headless Benchmark

Generated: `2026-03-11 16:55:15`

## Setup

- Mode: `headless` (no GUI, no animation render)
- Episodes per run: `50`
- Max steps per episode: `256`
- Repeats per combination: `5`

## Results

| Policy | Hidden Layers | Device | Repeats (ok/target) | Mean elapsed [s] | Std [s] |
|---|---|---|---:|---:|---:|
| PPO | `256` | CPU | 5/5 | 30.331 | 1.915 |
| PPO | `256,256` | CPU | 5/5 | 30.261 | 1.623 |
| SAC | `256` | CPU | 5/5 | 23.318 | 1.765 |
| SAC | `256,256` | CPU | 5/5 | 22.679 | 1.057 |
| TD3 | `256` | CPU | 5/5 | 9.649 | 0.387 |
| TD3 | `256,256` | CPU | 5/5 | 9.694 | 0.335 |

## Hidden Layer Impact

| Policy | Mean 256 [s] | Mean 256,256 [s] | Impact (256,256 vs 256) |
|---|---:|---:|---:|
| PPO | 30.331 | 30.261 | -0.2% |
| SAC | 23.318 | 22.679 | -2.7% |
| TD3 | 9.649 | 9.694 | +0.5% |
