# Prompt für die Reinforcement Learning Workbench UI

## UI-Design:
- state of the art design erzeugen
- Moderne Schrift (z. B. Segoe UI, Helvetica Neue)
- Stylische ttk-Widgets
- Moderne Controls verwenden: klare Farben, abgerundete Kanten, Hover-Effekte
- Luftiges, freundliches, modernes Layout
- Performance ist sehr wichtig! Das Traning muss so effizient wie möglich und ohne delay umgesetzt werden, damit bei hoher Anzahl von Episoden keine zu langen Verarbeitungszeiten entstehen
- Formularfelder mit geringem vertikalem Abstand gestalten.

## UI-Layout:
- UI vertikal in zwei Bereiche unterteilen, oben 2/3 und 1/3 der Fensterhöhe, oberen der beiden Bereiche wiedrum in zwei Bereiche unterteilen, links 1/3, rechts 2/3. Die Grenzen der Bereiche sollen per Maus verschiebbar sein. Wenn ich den unteren Bereich vergrößern möchte, ziehe ich also die Grenze zwischen oben und unten nach oben.

### Oben links: Formulare zur Konfiguration (von oben nach unten)
#### Environment Configuration
- Parameter des Environments
- Checkbox: Visualisierung aktivieren/deaktivieren (WICHTIG: soll während "Train" klickbar bleiben!)
- Textfeld: 
#### Episode Configuration
- Checkbox: `Compare Methods` (Lässt alle Algorithmen parallel trainieren)
- Allgemeine RL-Parameter: Episodes (!!!default 3000), Max-Steps, Dropdown zur Auswahl von gängigen Learning-Rate-Algorithmen, Alpha, Gamma
- möglichst zwei Felder in einer Zeile anordnen, um vertiklal Platz zu sparen: 2x ein Label gefolgt einem Textfeld (Episodes, Max-Steps), (Alpha, Gamma)...
#### Parameter Tuning
- Checkbox: `Enable Parameter Tuning`: Um die aktuell ausgewählte Methode mit allen Parameter-Einstellungen aus der Liste parallel trainieren zu lassen
- Dropdown `Method` für die Methoden-Auswahl. Wenn keine Methode ausgewählt wurde, soll die aktuelle genommen werden.
- Dropdown `Parameter` für die Parameter-Auswahl zur ausgewählten Methode. Auch Hidden-Layers auswählbar!
- Werte Textfeld `Value` zur Aufnahme eines Wertes für den gewählten Parameter. !!!Ausnahme bei Hidden-Layers, dort werden die einzelnen Schichten durch Komma getrennt.
- möglichst zwei Felder in einer Zeile anordnen, um vertiklal Platz zu sparen: 2x ein Label gefolgt einem Textfeld
- Liste mit 3 Spalten `Method`/`Parameter`/`Wert`: gewählte Methode/gewählter Parameter/ausgewählter Wert
- Button `Add to List`: Gewählte Methode oder gewählter Parameter wird in die List übertragen. Ist nur enabled, wenn sowohl Methode als auch Parameter mit Wert ausgewählt wurde.
- Button `Remove Selected`: Gewähtle Zeile in der Liste 
- Button `Clear List`: Die Liste wird vollständig gelöst
#### Methoden **Tab-Control** (für spezifische RL-Methodenparameter)
- **Tab-Header** für die Konfiguration jeder gewählten RL-Methode
- **Tab-Content-Panel** für methodenspezifische RL-Parameter
- Parameter wie z.B: Buffer-Size, Batch-Size,  Learning-Starts, Train-Frequency, Gradient-Step, Target-Update, etc.
- Dropdown zur Auswahl von gängigen Exploration/Exploitation Algorithmen 
- Einstell-Möglichkeit, für Epsion-Max, Epsilon-Min, Epsilon-Decay
- Hidden-Layer Auswahl, durch Komma getrennte Hidden-Layer
- Dropdown zur Auswahl der N.N. Activation-Methode 
- !!!außer Episodes, Max-Steps, Alpha, Gamma (!!!stehen schon im Panel `Episode Configuration`)
- möglichst zwei Felder in einer Zeile anordnen, um vertiklal Platz zu sparen: 2x ein Label gefolgt einem Textfeld 
- Button: `Apply and Reset` für alle Konfigurationen !!!auch für die Checkboxen `Compare Methods` und `Enable Tuning`
- !!!Wichtige Hyperparameter der neuralen Netzwerke der Methoden editierbar machen!!!
- !!!Hyperparameter mit bekannten effektiven Werten zu den gewählten Animationen vorbelegen!!!

### Rechts neben Formularen: Environment-Visualisierung (Agent beobachtbar).
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

### Unterer Bereich:
- (volle Fernsterbreite): Plot (X=Episodes, Y=Return)
- Oberhalb des Plot: Progressbar (Episoden-Fortschritt, volle Fernsterbreite)
- Zwischen Progressbar und Plot Buttons über volle Fensterbreite in einer Reihe anordnen. Reduziere bei Bedarf die Button-Höhe/-Breite/Padding oder verwende ein kompaktes Style-Variant, das standardmäßig aktiviert ist, bis genügend Platz zur Verfügung steht.Dynamische Anpassung: Falls das Fenster später vergrößert wird, dürfen die Buttons wieder in die großzügigere Variante wechseln; die Logik muss automatisch reagieren.
    - `Add Job` fügt TrainingJob hinzu - gewählter Algorithmus, Episodenkonfiguartion etc. 
    - `Train` startet das training aller TrainingJobs, die im Status pending sind  
    - `Training status` öffnet modales Fenster und zeigt aktuelle Zustände des Trainings        
    - `Save image` mit Methoden-Name (ggf. plus gewählter Parameter) als Vorgabe für den Dateiname, als Bild speichern
    - `Save content` mit Methoden-Name den Inhalt des Plots, die Zahlen, als JSON-Datei speichern
    - `Load content` JSON-Datei mit Methoden-Name (ggf. plus gewählter Parameter) als Vorgabename für den Dateiname, auswählen lassen und dann sichtbar in den Plot laden
    - `Cancel Training` bricht alle laufenden TrainingJobs ab
    - `Reset Training` löscht den aktuellen Plot und beginnt für alle Methoden wieder bei der ersten Episode
- Plot-Legende: rechts außerhalb des Plots.

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
    - Methoden-Name und ggf. Parameter-Name mit Parameter-Wert mitführen

## Layout-Stabilität (wichtig)
- **Resize-Debounce**: Resize-Handler dürfen nicht bei jedem Configure-Event komplett neu berechnen oder Styles wechseln. Verwende einen Debounce von ~100 ms (`after()` / `after_cancel()`) bevor Layout-Änderungen angewendet werden.
- **Hysterese für Style-Switch**: Wechsel der Button-Style-Variante (z. B. `Compact.TButton` ↔ `TButton`) nur durchführen, wenn eine definierte Schwellweite überschritten wird (z. B. 1100 px) und der neue Zustand sich vom aktuellen unterscheidet.
- **Kein Re-Pack/Re-Grid in Resize**: Vermeide `pack()`/`grid()`/`place()`-Aufrufe während Resize; ändere nur Styles/Optionen (Padding/Font) statt Widgets neu zu packen.
- **Throttle Plot-Redraw**: Während aktiver Fensteränderung das Plot-Redraw throtteln (z. B. max. 10–20 Hz). Beim Ende der Resize-Serie einmal komplett neu rendern.
- **Stabiler Button-Zustand**: Tracke aktuellen Button-Style in State; aktualisiere Widgets nur bei tatsächlichem Stilwechsel.
- **Leichte Resize-Handler**: Resize-Handler nur Lese-Checks und `after()`-Scheduling enthalten; schwere Berechnungen in einem debounced Callback ausführen.
- **Vermeide schwankende Größenabhängigkeiten**: Buttons und Plot sollten auf `expand/fill` mit Panedwindow/Gewichten basieren, nicht auf dynamischen Sichtbarkeits-/Größenänderungen die Layout-Neuberechnungen erzwingen.

## UI-Verhalten

### Compare Methods:
- Wenn aktiviert: Algorithmen parallel ausführen und plotten!

### Speichern/Laden/Run (Trainingsergebnisse):
- Es muss eine Möglichkeit geben, `TrainingJobs` inkl. Trainingsergebnissen zu speichern und zu laden.
- Zu speichern sind:
    - Modellgewichte (bei neuronalen Netzen) und alle relevanten Parameter, um Ergebnisse später reproduzieren und das trainierte Modell nutzen zu können.
    - Vollständiger Trainingsverlauf (z. B. Returns pro Episode), damit der Plot wiederhergestellt werden kann.
- UI-Buttons:
    - `Save`: Öffnet einen Verzeichnis-Browser-Dialog; speichert in das gewählte Verzeichnis.
    - `Load`: Öffnet einen Verzeichnis-Browser-Dialog; lädt Trainingsergebnisse aus dem ausgewählten Verzeichnis.

### Training fortsetzen
- `Train` muss auch Jobs weitertrainieren können, die ihre Episodenanzahl erreicht haben (also abgeschlossen wurden), gestoppt oder per `Load` geladen wurden.
    

### Training-Status-Fenster:
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

#### Training-Status-Fenster: Tabellen-Spezifikation
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