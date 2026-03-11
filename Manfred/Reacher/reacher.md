# RL-Workbench – Reacher-v5

## Voraussetzungen
- !!! WICHTIG !!!: Immer [workbench.md](../workbench.md), [workbench_ui.md](../workbench_ui.md) und [workbench_logic.md](../workbench_logic.md) berücksichtigen.
- Bitte Inhalte nicht einfach von vorherigen Versionen kopieren, es gibt Abweichungen!!
- Alle Regeln aus beiden Dateien gelten verbindlich.

## Spezielle Anforderungen

### Projekt
- `[projektname]` = `reacher`
- Ausgabeordner: `Manfred/Reacher/`
- Dateien: `reacher_app.py`, `reacher_logic.py`, `reacher_ui.py`

### Environment
- `gymnasium.make("Reacher-v5", render_mode="rgb_array")`
- Konfigurierbare Parameter (editierbar in Environment Configuration):
  - `reward_control_weight` (default `0.1`)
  - `reward_dist_weight` (default `1`)

### Algorithmen
- **SAC**, **TD3**, **PPO**,  via **Stable-Baselines3**
- Alle Hyperparameter einstellbar (inkl. Netzwerk-Architektur, Replay-Buffer etc.)

---