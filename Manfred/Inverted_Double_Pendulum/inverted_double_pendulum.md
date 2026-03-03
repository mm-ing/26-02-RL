# RL-Workbench – InvertedDoublePendulum-v5

## Voraussetzungen
- !!! WICHTIG !!!: Immer [workbench.md](../workbench.md) und [workbench_ui.md](../workbench_ui.md) berücksichtigen.
- Alle Regeln aus beiden Dateien gelten verbindlich.

## Spezielle Anforderungen

### Projekt
- `[projektname]` = `inverted_double_pendulum`
- Ausgabeordner: `Manfred/Inverted_Double_Pendulum/`
- Dateien: `inverted_double_pendulum_app.py`, `inverted_double_pendulum_logic.py`, `inverted_double_pendulum_ui.py`

### Environment
- `gymnasium.make("InvertedDoublePendulum-v5", render_mode="rgb_array")`
- Konfigurierbare Parameter (editierbar in Environment Configuration):
  - `healthy_reward` (default `10`)
  - `reset_noise_scale` (default `0.1`)

### Algorithmen
- **PPO**, **SAC**, **TD3** via **Stable-Baselines3**
- Alle Hyperparameter einstellbar (inkl. Netzwerk-Architektur, Replay-Buffer etc.)

---

## Implementierungsplan

### Schritt 1 – Logic-Layer (`inverted_double_pendulum_logic.py`)
- [ ] `EventType` Enum + `Event` Dataclass
- [ ] `EventBus` (Queue-basiert, thread-sicher)
- [ ] Config-Dataclasses: `PPOConfig`, `SACConfig`, `TD3Config`, `EnvConfig`, `EpisodeConfig`
- [ ] `WorkbenchCallback(BaseCallback)`: feuert Events pro Episode, prüft stop/pause-Events
- [ ] `JobStatus` Enum: PENDING, RUNNING, PAUSED, COMPLETED, CANCELLED, ERROR
- [ ] `TrainingJob`: job_id, algo_name, config, thread, stop/pause Events, Ergebnisse
- [ ] `TrainingManager`: add_job, start_job, pause, resume, cancel, remove → EventBus
- [ ] `CheckpointManager`: save/load (Modellgewichte + Trainingshistorie)

### Schritt 2 – UI-Layer (`inverted_double_pendulum_ui.py`)
- [ ] Dark-Theme-Setup (ttk-Styles, Farben nach workbench_ui.md)
- [ ] `WorkbenchApp(tk.Tk)`: Haupt-PanedWindow (oben 2/3 | unten 1/3)
  - [ ] Oben: horizontales PanedWindow (links 1/3 Config | rechts 2/3 Visualisierung)
  - [ ] Unten: Progressbar + Buttons + Matplotlib-Plot
- [ ] `ConfigPanel`: Environment Config, Episode Config, Parameter Tuning, Methods-TabControl
- [ ] `VisualizationPanel`: Canvas, Resize-Handler, Frame-Queue (10 ms Polling)
- [ ] `PlotPanel`: Matplotlib embedded, Moving-Avg, Raw, Dark-Theme-Farben
- [ ] `ButtonBar`: Add Job, Train, Training Status, Save/Load/Cancel/Reset
- [ ] `TrainingStatusWindow(tk.Toplevel)`: ttk.Treeview mit Live-Updates

### Schritt 3 – App-Einstieg (`inverted_double_pendulum_app.py`)
- [ ] `main()` erstellt `WorkbenchApp` und startet mainloop

### Schritt 4 – Tests (`tests/test_algorithms.py`)
- [ ] Unit-Tests für Logic-Layer (EventBus, TrainingManager, Config-Dataclasses)
- [ ] Simulations-Tests: Algo lernt auf InvertedDoublePendulum-v5 (kurzes Training, positiver Trend)

### Schritt 5 – Abhängigkeiten (`requirements.txt`)
- [ ] gymnasium[mujoco], stable-baselines3, torch, matplotlib, Pillow, numpy

### Schritt 6 – Validierung
- [ ] Tests ausführen, Fehler korrigieren
- [ ] App starten, Basis-Funktionalität prüfen (Add Job → Train → Plot aktualisiert sich)