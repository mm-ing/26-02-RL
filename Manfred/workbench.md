# Initial-Prompt für eine Reinforcement Learning Workbench mit verschiednenen Environments

## Regeln (verbindlich):
- Alle Prompts kurz, präzise, bevorzugt Stichpunkte.
- Folgeprompts klar benennen und fortlaufend nummerieren.
- Regeln in jedem Folgeprompt einhalten.

## Aufgabe 
- Lauffähiges Python-Programm (Workbench für Reinforcement Learning).
- Ausgabeordner: aktuelles Verzeichnis der [projektname].md
- Grundstruktur:
	- [projektname]_app.py
	- [projektname]_logic.py
	- [projektname]_ui.py

## UI-Design & UI-Layout & Layout-Stabilität & UI-Verhalten
- !!! WICHTIG !!!: Siehe [workbench_ui.md] berücksichtigen.

## Architektur: Schichten und Verantwortungen
### 1. `Environment` (Wrapper + Registry)
- Dünner Wrapper um gymnasium. Env-Name wird per Parameter übergeben, NICHT hardcoded.
- `EnvironmentRegistry` erlaubt Registrierung eigener Envs.
- Interface: `reset()`, `step(action)`, `render()`, `close()`, `observation_space`, `action_space`.

### 2. `ReplayBuffer`
- Zirkulärer Buffer für Off-Policy-Algorithmen.
- Eigenständig, wird vom Algorithmus intern instanziiert.
- Interface: `add(s, a, r, s', done)`, `sample(batch_size) -> TensorBatch`.

### 3. `AlgorithmBase` (Abstract) 
- Verantwortung: NUR Netzwerk-Architektur + Parameter-Updates.
- Interface:
  - `select_action(state, explore=True) -> action`
  - `update(batch) -> metrics_dict`  (einzelner Update-Schritt)
  - `get_state_dict() / load_state_dict()`  (Serialisierung)
- KEINE Trainingsschleife innerhalb des Algorithmus.
- Jeder Algorithmus hat eine zugehörige Config-Dataclass.
- Netzwerke werden via class `NetworkConfig` (hidden_layers, activation) konfiguriert.
  Factory-Funktion `build_mlp()` oder `build_cnn()` für einheitlichen Netzwerkaufbau.

### 4. `TrainLoop` (Trainingsschleife – SEPARATE Klasse)
- Orchestriert die Interaktion zwischen Algorithmus und Environment.
- Steuerbar von außen:
  - `run_episode() -> EpisodeResult`
  - `run_step() -> StepResult`
  - `run(n_episodes, callbacks) -> TrainingResult`
- Akzeptiert `stop_event` und `pause_event` für Thread-Steuerung.
- Feuert Events: `on_step`, `on_episode_end`, `on_training_done`.
- On-Policy (PPO, A2C, TRPO, A3C): sammelt Rollout, ruft dann `algo.update(rollout)`.
- Off-Policy (alle DQN, TD3, SAC, DDPG, ACER): nach jedem Step `algo.update(buffer.sample())`.

### 5. `TrainingJob`
- Repräsentiert einen einzelnen Trainings-Lauf.
- Hält: job_id, name, TrainLoop-Instanz, Thread, stop/pause Events,
  Ergebnis-Listen (returns, moving_avg), Sichtbarkeit (für UI).
- Kein UI-Code. Rein logische Verwaltungseinheit.

### 6. `TrainingManager` (Job-Scheduler)
- Verwaltet mehrere TrainingJobs.
- Interface: `add_job(config)`, `start_job(id)`, `start_all_pending()`,
  `pause(id)`, `resume(id)`, `cancel(id)`, `remove(id)`.
- Feuert Events an registrierte Listener (Observer-Pattern).
- Unterstützt: Compare-Mode (mehrere Algos), Tuning-Mode (Parameter-Sweep).
- Kein UI-Code. Kann headless genutzt werden (CLI, Tests, Notebooks).

### 7. `EventBus` / `UIBridge`
- Entkoppelt TrainingManager von UI.
- TrainingManager → EventBus → UI-Listener.
- Events: `JobCreated`, `EpisodeCompleted`, `StepCompleted`,
  `TrainingDone`, `FrameRendered`, `Error`.
- Thread-sicher (Queue-basiert für Tkinter).

### 8. `WorkbenchUI` (Tkinter/Web)
- NUR Darstellung und Benutzerinteraktion.
- Liest Events vom EventBus, aktualisiert Widgets.
- Delegiert alle Aktionen an TrainingManager.
- Unterteilt in Sub-Panels: ConfigPanel, PlotPanel, VisualizationPanel,
  StatusPanel (Treeview mit Jobs).

### 9. `CheckpointManager` + `MetricStore`
- Automatisches Speichern/Laden von Modellen und Trainingsverläufen.
- Strukturiertes Verzeichnisformat:
  `<experiment>/<job_id>/config.json, actor.pt, critic*.pt, metrics.json`

## Wichtige Design-Prinzipien
- Algorithmus enthält KEINE Trainingsschleife
- UI enthält KEINE Algorithmus-Logik
- TrainingManager ist UI-agnostisch
- Alle Kommunikation zwischen Threads über EventBus/Queue
- Environment-Name konfigurierbar, nicht hardcoded
- Agent-Klasse nur einführen wenn sie echten Zustand kapselt
  (z.B. Exploration-State, Episoden-Zähler), sonst weglassen


## Wichtige Regeln
- Wenn die Lösung neuronale Netzwerke beinhaltet:    
    - Nutze Pytorch aber NICHT Keras und Tensorflow! 
    - Anzahl der Hidden-Layer per Formular editierbar machen.
    - Anzahl der Neuronen pro Layer per Formular editierbar machen.
    - Layer mit unterschiedlich vielen Neuronen müssen definierbar sein.
    - In der UI neben Hidden-Layer/Neuronen auch weitere sinnvolle Einstellungen anbieten (z. B. Aktivierungsfunktion).
    - GPU nutzen, wenn sinnvoll und verfügbar.
    - Performance-Optimierungen anwenden:
        - `model.predict()` vermeiden; stattdessen `model(x, training=False)` nutzen.
        - Trainingsschritt als `@tf.function` kompilieren.
        - `float32`-Daten verwenden und implizite Casts vermeiden.
        - Training/Update in sinnvollen Batches durchführen (Overhead reduzieren).
        - Strikt vektorisieren: Numpy/Torch-Operationen statt Python-Loops, wo möglich.
        
## Asynchronität WICHTIG!
- Die App muss asynchron laufen. UI darf niemals einfrieren..
- Thread-sichere UI-Updates (z. B. Queue/`after()`), keine direkten UI-Zugriffe aus dem Trainings-Thread.
- Während asynchrones Training läuft: keine Monitoring-Schleifen im UI-Thread (z. B. `while working: sleep(2)`); stattdessen per Timer alle 10 ms Hintergrundstatus prüfen und UI aktualisieren.
- Plot, Progressbar und Environment-Visualisierung werden während des Trainings permanent aktualisiert.

## Performance
- Nach Implementierung der einzelnen Lern-Methoden, bitte mehrere Optimierungs-Runden einbauen zur Beschleunigung der Algorithmen und zur Verbesserung des Lernens

## Tests
- Schreibe Unit-Tests für die implementierten Methoden
- lege dafür ein Unterverzeichnis "test" im Ausgabeordner an, das die tests beinhalten soll 
- Führe nach Erzeugung des Codes die Tests aus und nimm ggf. Korrekturen am Code vor. Die Tests sollen nicht bei jedem Programm-Start ausgeführt werden
- Für jeden RL-Algorithmus in der Lösung: schreibe einen Simulationstest, der verifiziert, dass der Algorithmus korrekt arbeitet und lernt. Führe diese Tests aus und nimm bei Bedarf Korrekturen am getesteten Algorithmus vor.
- Prüfe anschließend nacheinander für jeden Reinforcement Learning Algorithmus einzeln:
    1. Ist die Implementierung des Algorithmus korrekt umgesetzt?
    2. Sind die Neuronalen Netze korrekt implementiert?
    3. Sind Trainingserfolge für jede Methode nachweisbar
    3. Sind Oprimierungen nötig, um einen Trainingserfolg zu erreichen oder ihn zu verbessern
    4. Prüfe, ob die UI alle Parameter des Algorithmus zum Editieren anbietet und mit sinnvollen defaults vorbelegt
    5. Nimm die sich aus 1., 2., 3. und 4. ergebenden Anpassungen vor
    6. Prüfe, ob nun Anpassungen an der UI erfoderlich geworden sind und nimm sie ggf. vor.