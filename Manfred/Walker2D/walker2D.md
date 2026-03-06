# RL-Workbench – Walker2d-v5

## Voraussetzungen
- !!! WICHTIG !!!: Immer [workbench.md](../workbench.md), [workbench_ui.md](../workbench_ui.md) und [workbench_logic.md](../workbench_logic.md) berücksichtigen.
- Alle Regeln aus beiden Dateien gelten verbindlich.

## Spezielle Anforderungen

### Projekt
- `[projektname]` = `walker2D`
- Ausgabeordner: `Manfred/Walker2D/`
- Dateien: `walker2D_app.py`, `walker2D_logic.py`, `walker2D_ui.py`

### Environment
- `gymnasium.make("Walker2d-v5", render_mode="rgb_array")`
- Konfigurierbare Parameter (editierbar in Environment Configuration):
  - `forward_reward_weight` (default `1`)
  - `ctrl_cost_weight` (default `0.001`)
  - `healthy_reward` (default `1`)
  - `terminate_when_unhealthy` (default `True`)
  - `healthy_z_range` (default `(0.8, 2)`)
  - `healthy_angle_range` (default `(-1, 1)`) 
  - `reset_noise_scale` (default `0.005`)
  - `exclude_current_positions_from_observation` (default `True`)

### Algorithmen
- **PPO**, **SAC**, **TD3** via **Stable-Baselines3**
- Alle Hyperparameter einstellbar (inkl. Netzwerk-Architektur, Replay-Buffer etc.)

---