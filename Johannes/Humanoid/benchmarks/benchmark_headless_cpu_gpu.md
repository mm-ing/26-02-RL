# Humanoid Headless Benchmark

Generated: `2026-03-12 09:24:54`

## Setup

- Mode: `headless` (no GUI, no animation render)
- Episodes per run: `50`
- Max steps per episode: `256`
- Repeats per combination: `5`

## Results

| Policy | Hidden Layers | Device | Repeats (ok/target) | Mean elapsed [s] | Std [s] |
|---|---|---|---:|---:|---:|
| PPO | `256` | CPU | 5/5 | 34.166 | 1.817 |
| PPO | `256` | GPU | 5/5 | 77.944 | 7.241 |
| PPO | `256,256` | CPU | 5/5 | 36.553 | 0.819 |
| PPO | `256,256` | GPU | 5/5 | 75.953 | 2.180 |
| SAC | `256` | CPU | 5/5 | 25.416 | 1.496 |
| SAC | `256` | GPU | 5/5 | 43.761 | 1.528 |
| SAC | `256,256` | CPU | 5/5 | 26.589 | 1.002 |
| SAC | `256,256` | GPU | 5/5 | 42.019 | 2.112 |
| TD3 | `256` | CPU | 5/5 | 10.810 | 0.660 |
| TD3 | `256` | GPU | 5/5 | 12.488 | 0.644 |
| TD3 | `256,256` | CPU | 4/5 | 11.290 | 0.392 |
| TD3 | `256,256` | GPU | 5/5 | 11.744 | 0.212 |

Note: For `TD3` with `hidden_layer=256,256` on `CPU`, repeat 1 (`21.12s`) was treated as an outlier and discarded. Reported mean/std and impact use repeats 2-5.

## Analysis

| Policy | Hidden Impact CPU (256,256 vs 256) | Hidden Impact GPU (256,256 vs 256) | GPU Influence @256 (GPU vs CPU) | GPU Influence @256,256 (GPU vs CPU) |
|---|---:|---:|---:|---:|
| PPO | +7.0% | -2.6% | +128.2% | +107.8% |
| SAC | +4.6% | -4.0% | +72.2% | +58.0% |
| TD3 | +4.4% | -6.0% | +15.5% | +4.0% |

Short discussion: In this headless setup, `GPU` is slower than `CPU` for all three policies at these benchmark scales, with the largest penalty on `PPO`, moderate on `SAC`, and smallest on `TD3`. Hidden-layer scaling is positive on CPU for all policies, while on GPU the `256,256` setting is slightly faster than `256` for `PPO`, `SAC`, and `TD3`.
