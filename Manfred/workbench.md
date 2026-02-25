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
- On-Policy (PPO): sammelt Rollout, ruft dann `algo.update(rollout)`.
- Off-Policy (TD3/SAC): nach jedem Step `algo.update(buffer.sample())`.

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

- Wenn die Lösung verschiedene Algorithmen vorsieht:
    - UI bietet je nach ausgewähltem Algorithmus nur die relevanten Einstellungen an.
    - Bei Änderung der Algorithmus-Auswahl wird das Formular dynamisch ausgetauscht und die passenden Einstellungen des neu gewählten Algorithmus angezeigt.

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

## Design:
- state of the art design erzeugen
- Moderne Schrift (z. B. Segoe UI, Helvetica Neue)
- Stylische ttk-Widgets
- Moderne Controls verwenden: klare Farben, abgerundete Kanten, Hover-Effekte
- Luftiges, freundliches, modernes Layout
- Performance ist sehr wichtig! Das Traning muss so effizient wie möglich und ohne delay umgesetzt werden, damit bei hoher Anzahl von Episoden keine zu langen verarbeitungszeiten entstehen
- Formularfelder mit geringem vertikalem Abstand gestalten.

## Asynchronität WICHTIG!
- Die App muss asynchron laufen. UI darf niemals einfrieren..
- Thread-sichere UI-Updates (z. B. Queue/`after()`), keine direkten UI-Zugriffe aus dem Trainings-Thread.
- Während asynchrones Training läuft: keine Monitoring-Schleifen im UI-Thread (z. B. `while working: sleep(2)`); stattdessen per Timer alle 10 ms Hintergrundstatus prüfen und UI aktualisieren.
- Plot, Progressbar und Environment-Visualisierung werden während des Trainings permanent aktualisiert.

## UI-Layout:
- UI vertikal in zwei Bereiche unterteilen, oben 2/3 und 1/3 der Fensterhöhe, oberen der beiden Bereiche wiedrum in zwei Bereiche unterteilen, links 1/3, rechts 2/3. Die Grenzen der Bereiche sollen per Maus verschiebbar sein. Wenn ich den unteren Bereich vergrößern möchte, ziehe ich also die Grenze zwischen oben und unten nach oben.
- Oben links: Formulare zur Konfiguration
    - Environment Configuration
        - Parameter des Environments
        - Checkbox: Visualisierung aktivieren/deaktivieren (WICHTIG: soll während "Train" klickbar bleiben!)
        - Textfeld: 
    - Episode Configuration
        - Checkbox: Compare Methods (Lässt alle Algorithmen parallel trainieren)
        - Allgemeine RL-Parameter: Episodes, Max-Steps, Alpha, Gamma, Policy, Pathfinder etc.
        - möglichst zwei Felder in einer Zeile anordnen, um vertiklal Platz zu sparen: ein Label gefolgt von zwei Textfeldern (Episodes, Max-Steps), (Alpha, Gamma)...
    - Methoden Configration
        - Tab für jede einzelne Methode
        - Methodenspezifische RL-Parameter: Buffer-Size, Target-Update etc...
        - möglichst zwei Felder in einer Zeile anordnen, um vertiklal Platz zu sparen: ein Label gefolgt von zwei Textfeldern 
    - Button: Apply and reset für alle Konfigurationen   

- Rechts neben Formularen: Environment-Visualisierung (Agent beobachtbar).
	- Die Darstellung MUSS den vorgesehenen Bereich des Fensters ausfüllen, ohne Verzerrung:
		- Bild/Canvas dynamisch an das Widget-Layout anpassen (Resize-Handler).
		- Seitenverhältnis der Gymnasium-Grafik strikt beibehalten (keine Streckung).
		- Maximale Skalierung innerhalb des Containers (Letterboxing nur wenn nötig, zentriert).
		- Container soll expand/fill nutzen, damit die Visualisierung den Platz tatsächlich erhält.
    - Keine Warteschlange von Frames aufbauen; keine Abarbeitung nach Trainingsende.
    - Alle 10 ms den aktuellsten Frame darstellen; alle anderen Frames verwerfen.
    - Intervall (10 ms) muss editierbar sein (Textfeld in Environment Configuration).  
    - Konfiguration für Environment:
        - Wenn das Environment Konfigurationsmöglichkeiten bietet, MÜSSEN diese in dem Formular Environment Configuration editierbar sein. 
- Unterer Bereich:
    - (volle Fernsterbreite): Plot (X=Episodes, Y=Return)
    - Oberhalb des Plot: Progressbar (Episoden-Fortschritt, volle Fernsterbreite)
    - Zwischen Progressbar und Plot Buttons über volle Fensterbreite in einer Reihe anordnen. Reduziere bei Bedarf die Button-Höhe/-Breite/Padding oder verwende ein kompaktes Style-Variant, das standardmäßig aktiviert ist, bis genügend Platz zur Verfügung steht.Dynamische Anpassung: Falls das Fenster später vergrößert wird, dürfen die Buttons wieder in die großzügigere Variante wechseln; die Logik muss automatisch reagieren.
        - `Add Job` (Fügt TrainingJob hinzu - gewählter Algorithmus, Episodenkonfiguartion etc. )
        - `Train` (startet das training aller TrainingJobs, die im Status pending sind)          
        - `Save plot` (als Bild speichern)
        - `Cancel Training` (bricht alle laufenden 
        TrainingJobs ab)
        - `Reset Training` löscht den aktuellen Plot und beginnt für alle Methoden wieder bei der ersten Episode
    - Plot-Legende: standardmäßig oben rechts; wenn die ersten 4 episoden überschitten werden, nach unten links verschieben.

## Layout-Stabilität (wichtig)
- **Resize-Debounce**: Resize-Handler dürfen nicht bei jedem Configure-Event komplett neu berechnen oder Styles wechseln. Verwende einen Debounce von ~100 ms (`after()` / `after_cancel()`) bevor Layout-Änderungen angewendet werden.
- **Hysterese für Style-Switch**: Wechsel der Button-Style-Variante (z. B. `Compact.TButton` ↔ `TButton`) nur durchführen, wenn eine definierte Schwellweite überschritten wird (z. B. 1100 px) und der neue Zustand sich vom aktuellen unterscheidet.
- **Kein Re-Pack/Re-Grid in Resize**: Vermeide `pack()`/`grid()`/`place()`-Aufrufe während Resize; ändere nur Styles/Optionen (Padding/Font) statt Widgets neu zu packen.
- **Throttle Plot-Redraw**: Während aktiver Fensteränderung das Plot-Redraw throtteln (z. B. max. 10–20 Hz). Beim Ende der Resize-Serie einmal komplett neu rendern.
- **Stabiler Button-Zustand**: Tracke aktuellen Button-Style in State; aktualisiere Widgets nur bei tatsächlichem Stilwechsel.
- **Leichte Resize-Handler**: Resize-Handler nur Lese-Checks und `after()`-Scheduling enthalten; schwere Berechnungen in einem debounced Callback ausführen.
- **Vermeide schwankende Größenabhängigkeiten**: Buttons und Plot sollten auf `expand/fill` mit Panedwindow/Gewichten basieren, nicht auf dynamischen Sichtbarkeits-/Größenänderungen die Layout-Neuberechnungen erzwingen.

### Neural Networks
- !!!Wichtige Hyperparameter der neuralen Netzwerke der Methoden editierbar machen!!!
- Hyperparameter mit bekannten effektiven Werten vorbelegen

## Plot-Anforderungen:
- Moving-Average als dicke Linie im Vordergrund
- Rohdaten als dünne Linie, jede Episode (Raw) im Hintergrund
- Plot-Farbgestaltung exakt wie folgt umsetzen:
    - Figure-Hintergrund: #0f111a
    - Axes-Hintergrund: #0f111a
    - Tick-Farben: #b5b5b5
    - Achsen-Labels (X/Y): #b5b5b5
    - Grid: Farbe #2a2f3a, gestrichelt, Alpha 0.5
    - Raw-Line: Farbe #4cc9f0, Alpha 0.35, Linienbreite 1.0 (nur für die erste Linie)
    - Moving-Average-Line: Farbe #4cc9f0, Alpha 1.0, Linienbreite 2.5 (nur für die erste Linie)
    - Weitere Linien: gleicher Stil (Raw: Alpha 0.35, LW 1.0; Avg: Alpha 1.0, LW 2.5), aber mit gutem Kontrast in anderen Farben
    - Legend: Facecolor #0f111a, Edgecolor #2a2f3a, Labelcolor #e6e6e6


## Compare Methods:
- Wenn aktiviert: Algorithmen parallel ausführen und plotten

## Workbench (Vergleich & Parameter-Tuning):
- Die Lösung soll eine Workbench für den Vergleich verschiedener RL-Algorithmen und das Parameter-Tuning eines einzelnen Algorithmus sein.
- Neben "Compare Methods" (Algorithmen parallel vergleichen) muss es einen Modus für Parameter-Tuning eines einzelnen Algorithmus geben.
- Parameter-Tuning: Ein ausgewählter Parameter (z. B. `alpha`) wird mit mehreren Werten parallel getestet und im Plot verglichen.
- UI-Anforderung für Parameter-Tuning (Formular):
    - Auswahl des zu tunenden Parameters
    - Min-Wert, Max-Wert, Step (Schrittweite)
    - Aus Min/Max/Step ergibt sich eine Werteliste; für jeden Wert wird eine eigene Variante parallel trainiert.
    - Beispiel: Min=2, Max=6, Step=2 → Variants mit 2, 4, 6; alle laufen parallel und werden im Plot verglichen.

## Speichern/Laden/Run (Trainingsergebnisse):
- Es muss eine Möglichkeit geben, `TrainingJobs` inkl. Trainingsergebnissen zu speichern und zu laden.
- Zu speichern sind:
    - Modellgewichte (bei neuronalen Netzen) und alle relevanten Parameter, um Ergebnisse später reproduzieren und das trainierte Modell nutzen zu können.
    - Vollständiger Trainingsverlauf (z. B. Returns pro Episode), damit der Plot wiederhergestellt werden kann.
- UI-Buttons:
    - `Save`: Öffnet einen Verzeichnis-Browser-Dialog; speichert in das gewählte Verzeichnis.
    - `Load`: Öffnet einen Verzeichnis-Browser-Dialog; lädt Trainingsergebnisse aus dem ausgewählten Verzeichnis.

## Training fortsetzen
- `Train` muss auch Jobs weitertrainieren können, die ihre Episodenanzahl erreicht haben (also abgeschlossen wurden), gestoppt oder per `Load` geladen wurden.
    

## Training-Status-Fenster:
- Neben "Train" einen Button hinzufügen, der ein neues Fenster öffnet.
- Das Fenster zeigt für die `Jobs` (Instanzen von `TrainingJob`) übersichtlich:
    - Episode x von y
    - Wichtige Kennzahlen (Trainingserfolg/-Verlauf), z. B. Return, Moving-Average, Epsilon, Loss
    - Dauer pro Episode und Schritte
    - Live-Updates während des Trainings (thread-sicher via Queue/`after()`)
    - Pro TrainingJob Steuerung:
        - Sichtbarkeit im Plot umschalten (ein-/ausblenden)
        - `Train`: starten/fortsetzen des Trainings 
        - `Run`: startet den selektierten TrainingJob ohne zu trainieren (nur Ausführen/Validieren) nur enabled, wenn TrainingJob zuvor trainiert wurde
        - `Stop`: beendet TrainingJob, Button ist nur enabled wenn TrainingJob aktiv (Train oder Run)
        - `Remove`: Löscht den selektierten TrainingJob. Wenn der Job gerade läuft, muss er vorher sauber gestoppt werden. Stelle sicher, dass `Remove` nicht nur den Eintrag im Training-Status-Fenster entfernt, sondern auch den Algorithmus und Agent und ggf. der zugehörige Hintergrund-Thread, die durch diesen Eintrag visualisiert werden. 
- Wenn es mehrere aktive TrainingJobs gibt, soll die Visualisierung im Hauptfenster immer das Environment des laufenden Algorithmus anzeigen, der im „Training Status“-Fenster aktuell selektiert ist (sofern eine Auswahl vorliegt).
- Wenn keine Auswahl vorliegt, soll der erste laufende Job visualisiert werden.
- Thread-sichere UI-Updates beibehalten.
       


### Training-Status-Fenster: Tabellen-Spezifikation
- **Verwende ein modernes Tabellen-Widget** (z. B. `ttk.Treeview`) — keine individuellen Label-/Button-Zeilen pro TrainingJob. Tabelle muss performant scrollen und Spalten anpassen.
- **Spalten (Pflicht):** `Algorithm`, `Episode` (x/y), `Return`, `MovingAvg`, `Epsilon`, `Loss`, `Duration`, `Steps`, `Visible`.
- **Visibility**: `Visible` als text/bool in der Tabelle darstellen; Doppelklick auf Zeile oder Button außerhalb der Tabelle toggelt Sichtbarkeit und löst `on_toggle_visibility(alg, visible)` aus.
- **Zeilen-Selektion & Aktionen:** unterhalb/seitlich der Tabelle sind globale Aktions-Buttons für die selektierte Zeile: `Toggle Visibility`, `Pause`, `Resume`, `Cancel`, `Restart`. Aktionen wirken auf die aktuell selektierte TrainingJob-Zeile.
- **Inline-Interaktion:** Doppelklick auf eine Zeile toggelt Visibility; Kontextmenü (Rechtsklick) bietet dieselben Aktionen an.
- **Live-Updates:** UI erhält Updates ausschließlich über eine Queue; beim Eintreffen von Daten (`episode_end`, `progress`, `training_done`) wird die jeweilige Tabellenzeile aktualisiert. Tabelle nur per `item()` updaten, keine komplette Neuaufbau-Operationen.
- **Thread-Sicherheit:** Keine direkten UI-Änderungen aus dem Trainings-Thread; alle Änderungen über `after()`/Queue erfolgen.
- **Sortierbarkeit & Breite:** Spalten sollten sortierbar sein (klick auf Header). Standard-Breiten setzen, `algorithm` mit `stretch=True`.
- **Wenige Widgets pro Zeile:** Vermeide Widgets pro Zelle (z. B. Checkbox-Widgets). Stattdessen verwende Text-/Symbolrepräsentation und zentrale Steuerbuttons.
- **Accessibility & Keyboard:** Tab/Up/Down wählbar; Enter = Toggle Visibility; Space = Pause/Resume (konfigurierbar).
- **Styling:** Tabellen-Hintergrund und Kopfzeile im App-Theme (#0f111a / #2a2f3a); Zeilen-Hover/Selection-Farbe deutlich kontrastreich.
- **Performance:** Bei hoher Update-Frequenz nur betroffene Zeile aktualisieren; Rate-limit Updates für dieselbe Zeile (z. B. max 20 Hz) um UI-Load zu begrenzen.
- **Persistenz (optional):** Auswahl (Visible/Paused) bleibt erhalten, wenn das Fenster geschlossen und wieder geöffnet wird (lokaler State-Cache im App-State).

Diese Regeln sind verpflichtend für die Implementierung des Training-Status-Fensters.

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