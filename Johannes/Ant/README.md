# Ant Project

`Ant` training GUI for reinforcement and evolutionary learning on `Ant-v5`.

## Backends
- Reinforcement learning: Stable-Baselines3 (`SAC`) and sb3-contrib (`TQC`)
- Evolutionary learning: EvoTorch (`CMA-ES`, with safe fallback when EvoTorch is unavailable)

## Run
```bash
python Ant_app.py
```

## Tests
```bash
python -m pytest -q --rootdir . --confcutdir . tests
```

## Output
- CSV samplings: `results_csv/`
- Plot PNGs: `plots/`
