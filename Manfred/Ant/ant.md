# RL-Workbench – Ant-v5

## Voraussetzungen
- !!! WICHTIG !!!: Immer [workbench.md](../workbench.md), [workbench_ui.md](../workbench_ui.md) und [workbench_logic.md](../workbench_logic.md) berücksichtigen.
- Bitte Inhalte nicht einfach von vorherigen Versionen kopieren, es gibt Abweichungen!!
- Alle Regeln aus beiden Dateien gelten verbindlich.

## Spezielle Anforderungen

### Projekt
- `[projektname]` = `ant`
- Ausgabeordner: `Manfred/Ant/`
- Dateien: `ant_app.py`, `ant_logic.py`, `ant_ui.py`

### Environment
- `gymnasium.make("Ant-v5", render_mode="rgb_array")`
- Konfigurierbare Parameter (editierbar in Environment Configuration):
  - `forward_reward_weight` (default `1`)
  - `ctrl_cost_weight` (default `0.5`)
  - `contact_cost_weight` (default `5e-4`)
  - `healthy_rewardmain_body` (default `1(“torso”)`)
  - `terminate_when_unhealthy` (default `True`)
  - `healthy_z_range` (default `0.2, 1`)
  - `contact_force_range` (default `-1, 1`)
  - `reset_noise_scale` (default `1`)
  - `exclude_current_positions_from_observation` (default `True`)
  - `include_cfrc_ext_in_observation` (default `True`)
  - `use_contact_forces (v4 only)` (default `False`)

### Algorithmen
- **TQC**, **SAC** via **Stable-Baselines3** und **CMA-ES** mit EvoTorch 
- Alle Hyperparameter einstellbar (inkl. Netzwerk-Architektur, Replay-Buffer etc.)

---