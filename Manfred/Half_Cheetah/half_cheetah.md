# RL-Workbench – HalfCheetah-v5

## Voraussetzungen
- !!! WICHTIG !!!: Immer [workbench.md](../workbench.md) und [workbench_ui.md](../workbench_ui.md) berücksichtigen.
- Alle Regeln aus beiden Dateien gelten verbindlich.

## Spezielle Anforderungen

### Projekt
- `[projektname]` = `half_cheetah`
- Ausgabeordner: `Manfred/Half_Cheetah/`
- Dateien: `half_cheetah_app.py`, `half_cheetah_logic.py`, `half_cheetah_ui.py`

### Environment
- `gymnasium.make("HalfCheetah-v5", render_mode="rgb_array")`
- Konfigurierbare Parameter (editierbar in Environment Configuration):
  - `forward_reward_weight` (default `1`)
  - `ctrl_cost_weight` (default `0.1`)
  - `reset_noise_scale` (default `0.1`)
  - `exclude_current_positions_from_observation` (default `True`)

### Algorithmen
- **PPO**, **SAC**, **TD3** via **Stable-Baselines3**
- Alle Hyperparameter einstellbar (inkl. Netzwerk-Architektur, Replay-Buffer etc.)

---

