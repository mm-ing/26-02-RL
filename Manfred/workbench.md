# Initial-Prompt für eine Reinforcement Learning Workbench mit einstellbarem Environment

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

## Architektur der Logik: Schichten und Verantwortungen
- !!! WICHTIG !!!: Siehe [workbench_logic.md] berücksichtigen.
        
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