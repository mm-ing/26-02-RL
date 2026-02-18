Du bist GitHub Copilot (GPT-5 mini). Lies die Datei vollständig und erstelle knapp und strukturiert in Deutsch:
1) Erzeuge eine fertige Datei bandit.py im selben Ordner mit Tkinter‑GUI und folgenden Features:
   - Drei einarmige Banditen (Slots).
   - Starten kostet 1 Coin pro Spiel.
   - Jeder Automat gibt eine zufällige Anzahl Coins zurück; die erwartete Auszahlung steigt mit der Anzahl Coins, die bereits im Automaten sind.
   - Initialisiere die Automaten mit drei unterschiedlichen Startmengen (20, 40, 80).
   - Berechne die Wahrscheinlichkeit für die Ausgabe von Coins gemäß der Formel: bereits eingeführte Coins/100
   - GUI: 3 Buttons (1 button je Automat). Button drücken wirft 1 Coin ein und startet das Spiel.
   - Über den Buttons steht der Text: "Glücksspiel kann süchtig machen!"
   - Für jeden Button zeige: Anzahl Versuche (Clicks) und Anzahl zurückgegebener Coins (gesamt).
2) Liefere zusätzlich:
   - Kurze Zusammenfassung (2–3 Sätze).
   - Implementierungsschritte (nummeriert), Priorität (hoch/mittel/niedrig) und geschätzter Aufwand in Stunden.
   - Minimale Projektstruktur + Code‑Stubs (Dateinamen + kurze Inhalte) für bandit.py und evtl. tests.
   - Pytest Unit‑Test‑Skizzen für die Logik (nicht GUI).
   - Beispielbefehl zum Starten unter Windows PowerShell.
3) Liste maximal 5 gezielte Rückfragen, falls Informationen fehlen (z. B. genaue Startmengen, Auszahlungspolicy).
Antworte knapp, nutze Codeblöcke für Code (Python/PowerShell), keine langen Erklärungen.


# Reinforcement
Überführe den aktuellen Stand des Projekts in ein Reinforcement Projekt. Berücksichte dafür folgende Requirements:
- Ziel: der Agent soll herausfinden von welchem Automaten die meisten Coins ausgeworfen werden.
- Erstelle drei Dateien: bandit_gui.py; bandit_app.py; bandit_logic.py.
- Inhalt bandit_app.py:
   - entry point für die Initialisierung der anderen Klassen
- Inhalt bandit_gui.py:
   - Klasse GUI
   - Tkinter GUI basierend auf der bestehenden GUI
   - erweitere die GUI um die Buttons: Agent single step; Agent run n loops; Reset
   - füge Plots (matplotlib) hinzu: cumulative reward; Legende für Policy; unterschiedliche Farbe für unterschiedliche policies; live plot
- Inhalt bandit_logic.py:
   - Klasse "bandit":
      - einarmiger Bandit gemäß des bestehenden codes
      - eine Instanz pro Bandit
   - Klasse "environment":
      - erstellt die Welt, bestehend aus den drei einarmigen Banditen
   - Klasse "agent":
      - Agent, der die Spielautomaten nutzt
      - action: ausführen eines Automaten
      - reward: vom Automaten ausgeworfene Coins
   - Klasse "policy":
      - bestimmt die policy für das Training des Agenten
- Agent Loops (n) = 100
- Agent Memory (0=all) = 0
- Epsilon = 0.9; Decay = 0.01
- Policy: Epsilon-Greedy; Thompson Sampling
- Current state: bandit; pulls; success; success_rate

# GPT Prompt
Implementiere ein kleines Reinforcement-Projekt in Deutsch. Erzeuge diese Dateien in demselben Ordner: `bandit_logic.py`, `bandit_app.py`, `bandit_gui.py`. Ziele, Anforderungen und Deliverables:

1) `bandit_logic.py`
 - Klassen: `Bandit` (einarmiger Bandit, Startbestand konfigurierbar, Auszahlung gemäß vorhandener Coins), `Environment` (3 Bandits initialisieren, Schnittstellen: step(action)->reward, reset()), `Agent` (Speichert Aktionen, Rewards, Action-Methode: wähle Bandit und execute), `Policy` (implementiere `EpsilonGreedy` und `ThompsonSampling`).
 - Agent-Parameter: loops n=100, memory=0 (0=unbegrenzt), epsilon_start=0.9, epsilon_decay=0.01.
 - State fields per Bandit: `bandit`, `pulls`, `success`, `success_rate`.
 - API: klar testbare Logik-Funktionen (no-GUI).

2) `bandit_app.py`
 - Entry point: initialisiere `Environment`, `Agent` und `GUI`, binde Aktionen.
 - CLI: optionales Start-Argument `--runs n`.

3) `bandit_gui.py`
 - Klasse `GUI` mit Tkinter: Anzeige ähnlich bestehender GUI + Buttons: `Agent single step`, `Agent run n loops`, `Reset`.
 - Zeige Warntext: "Glücksspiel kann süchtig machen!".
 - Live-Plots mit `matplotlib` embedded: cumulative reward; Legende für Policy; unterschiedliche Farbe für jede Policy; live update während Runs.
 - Statusanzeigen: pulls und cumulative reward pro Bandit, Policy-Farbe-Legende.
 - Buttons für manuelle Spielzüge (wie bisher).

4) Tests & Run
 - Schreibe Pytest-Unit-Test-Skizzen für Logik in `tests/test_bandit_logic.py` (Test: Environment.step(), Agent bei EpsilonGreedy wählt Actions, ThompsonSampling posterior updates).
 - Beispiel-PowerShell-Befehl zum Starten GUI: `python .\bandit_app.py`
 - Requirements: `tkinter`, `matplotlib`, `numpy`

5) Weitere Anforderungen/Verhalten
 - Auszahlungspolicy: Wahrscheinlichkeit für Auszahlung = already_inserted_coins / 100 (capped at 1.0). Bei Auszahlung: wähle zufälligen Betrag <= aktuellem Bestand.
 - Initiale Startmengen: 20, 40, 80.
 - Agent Loops default = 100, Memory default = 0, Epsilon start 0.9, Decay 0.01.
 - Pro Run: aktualisiere live-plot cumulative reward; nach Run zeige finale Statistik welche Maschine die meisten Coins auswarf.

6) Deliverables (konkret)
 - Implementierte Dateien: `bandit_logic.py`, `bandit_app.py`, `bandit_gui.py`
 - `tests/test_bandit_logic.py` mit 3-4 pytest-Skizzen
 - Kurze README-Anweisung (1-2 Zeilen) mit Start-Command

Antworte mit: erstelle die Dateien, laufbare Implementierung und Tests; wenn Informationen fehlen, stelle bis zu 5 kurze Rückfragen.

# Implemented
- Dateien erstellt: `bandit_logic.py`, `bandit_gui.py`, `bandit_app.py`
- Tests erstellt: `tests/test_bandit_logic.py` (4 Tests)
- Zusatzdateien: `README.md`, `requirements.txt`

# How to run
```powershell
python .\bandit_app.py
```

```powershell
pytest .\tests\test_bandit_logic.py
```

# Standard-Parameter
- Agent Loops (default): 100
- Agent Memory (0 = unbegrenzt): 0
- Epsilon Start: 0.9
- Epsilon Decay: 0.01
- Startmengen Bandits: 20, 40, 80
- Policy-Optionen: Epsilon-Greedy, Thompson Sampling

| Parameter | Bedeutung |
|---|---|
| Agent Loops | Anzahl Agent-Schritte pro Run (Default: 100) |
| Agent Memory | Anzahl gespeicherter letzter Schritte, `0` = unbegrenzt |
| Epsilon Start | Initiale Explorationsrate der Epsilon-Greedy-Policy |
| Epsilon Decay | Multiplikative Abnahme von Epsilon pro Schritt |
| Startmengen Bandits | Initiale Coin-Bestände: 20, 40, 80 |
| Policy-Optionen | Verfügbare Policies: Epsilon-Greedy, Thompson Sampling |