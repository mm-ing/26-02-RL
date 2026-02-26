# RL-Workbench reinforcement learning für gymnasium cartpole

## Vorrausetzungen:
- !!! WICHTIG !!!: Immer [Workbench.md](../Workbench.md) berücksichtigen.
	- Alle Regeln aus [Workbench.md](../Workbench.md) gelten verbindlich.
	- Prompt/Arbeitsweise strikt nach diesen Regeln.
    - Führe die Anweisungen in [Workbench.md](../Workbench.md) aus

## Spezielle Anforderungen

### Projekt
- [projektname] = `cartpole`

### Animation
- Nutze Environment `gymnasium.make("CartPole-v1")`
- Die Environment-Visualisierung soll die Gymnasium-`CartPole-v1` die animierte grafische Ausgabe.
- Parameter
    - goal_velocity (default 0.1) 
    - x_init (default -0.6)
    - y_init (default -0.4)
     
### Algorithmen (auswählbar):
- DDQN (double dqn)
- Dueling DQN
- Double + Dueling DQN (D3QN)
- Nutze Stable-Baselines3
- Alle Hyperparameter der einzelnen Methoden, wie z.B. der Relay-Buffer als auch die Hyperparameter der neuronalen Netze sollen einstellbar sein. 

---

## README

### Überblick
- RL-Workbench für `CartPole-v1` mit asynchronem Training und Live-Visualisierung.
- Vergleichsmodus (`Compare Methods`) für paralleles Training mehrerer Methoden.
- Parameter-Tuning-Modus für einen frei wählbaren Hyperparameter mit `Min/Max/Step`.
- Persistenz von Trainingsjobs inkl. Modell und Metriken (`Save`/`Load`).

### Projektstruktur
- `cartpole_app.py` – Startpunkt der Anwendung
- `cartpole_ui.py` – Tkinter-UI, Plot, Status-Fenster, Event-Polling
- `cartpole_logic.py` – TrainingManager, TrainLoop, Jobs, EventBus, Checkpoints
- `requirements.txt` – Abhängigkeiten
- `test/test_cartpole_logic.py` – Unit- und Simulationstests

### Installation
1. In den Ordner wechseln:
	 - `cd Manfred/Cartpole`
2. (Optional) Virtuelle Umgebung aktivieren.
3. Abhängigkeiten installieren:
	 - `python -m pip install -r requirements.txt`

### Start
- Anwendung starten:
	- `python cartpole_app.py`

### Bedienung (Kurz)
1. Environment konfigurieren (`CartPole-v1`, Visualisierung, Frame-Intervall).
2. Episoden- und Algorithmus-Parameter setzen.
3. Optional:
	 - `Compare Methods` aktivieren für DDQN, Dueling DQN und D3QN parallel.
	 - `Enable Tuning` aktivieren und Parameterbereich definieren.
4. `Add Job` klicken.
5. `Train` klicken.
6. Status über `Training Status` überwachen (Pause/Resume/Cancel/Visibility).
7. Ergebnisse mit `Save plot` oder komplette Jobs mit `Save` speichern.

### Plot- und Visualisierungsdetails
- Plot zeigt pro sichtbarem Job:
	- Raw-Returns (dünn, transparent)
	- Moving-Average (dick, im Vordergrund)
- Legende:
	- Standard oben rechts
	- Wechsel nach unten links, sobald ein Job > 4 Episoden hat
- Environment-Frame wird thread-sicher aktualisiert; nur neuester Frame wird angezeigt.

### Training fortsetzen
- `Train` startet auch Jobs neu/weiter, die bereits beendet, gestoppt oder geladen wurden.

### Tests
- Tests ausführen:
	- `python -m pytest -q`
- Enthalten:
	- Unit-Tests für Registry, ReplayBuffer, Manager und Persistenz
	- Simulationstests für DDQN, Dueling DQN und D3QN

### Hinweise
- Für die Implementierung wird `Stable-Baselines3` genutzt.
- Die drei auswählbaren Varianten werden über unterschiedliche DQN-Konfigurationsprofile abgebildet.