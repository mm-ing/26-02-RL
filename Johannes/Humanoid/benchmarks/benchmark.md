# Humanoid Benchmark

Generated: `2026-03-11 16:41:37`

## Setup

- Environment: `Humanoid-v5`
- Execution path: full `HumanoidGUI` training flow with animation enabled
- Episodes per run: `1`
- Max steps per episode: `64`
- Animation: `on`, `fps=30`, `frame_stride=1`, `update_rate=1`
- Repeats per combination: `5`
- GPU available: `True`

## Run Summary

- Total runs scheduled: `20`
- Successful runs: `20`
- Skipped runs: `0`
- Error runs: `0`

## Results (Mean and Std)

| Policy | Hidden Layers | Device | Repeats (ok/target) | Mean elapsed [s] | Std [s] |
|---|---|---|---:|---:|---:|
| PPO | `256` | CPU | 5/5 | 3.220 | 0.199 |
| PPO | `256,256` | CPU | 5/5 | 3.711 | 0.290 |
| SAC | `256` | CPU | 5/5 | 2.322 | 0.268 |
| SAC | `256,256` | CPU | 5/5 | 2.173 | 0.299 |
| TD3 | `256` | CPU | 0/5 | - | - |
| TD3 | `256,256` | CPU | 0/5 | - | - |

## Hidden Layer Impact

| Policy | Device | Mean 256 [s] | Mean 256,256 [s] | Impact (256,256 vs 256) |
|---|---|---:|---:|---:|
| PPO | CPU | 3.220 | 3.711 | +15.3% |
| SAC | CPU | 2.322 | 2.173 | -6.4% |
| TD3 | CPU | - | - | - |

## Bottleneck Analysis

- Changing hidden layers from `256` to `256,256` changed elapsed time by about `4.4%` on average.
- Across all tested settings, `SAC` was fastest and `PPO` slowest, with a mean-gap of about `54.2%` in elapsed time.
- Interpretation focus: this setup includes full Tkinter + Matplotlib updates and frame playback, so measured time combines RL compute, environment simulation, and GUI animation costs.
