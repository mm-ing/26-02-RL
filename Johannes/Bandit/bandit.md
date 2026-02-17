Erstelle ein skript bandit.py in diesem Ordner mit folgenden Feature:
- simuliere drei Glüccksspielautomaten (einarmiger Bandit)
- starten des Automats kostet jeweils 1 Coin 
- jeder Automat soll eine zufällige Anzahl Coins ausgeben, wobei mehr Coins ausgegeben werden, desto mehr Coins bereits im Automaten sind
- initialisiere die AUtomaten mit 3 unterschiedlichen Startmengen bereits vorhandener Coins
- erzeuge ein tkinter user interface mit 3 buttons, ein button pro Automat, das drücken des Buttons wirft einen Coin in den Automaten und startet das Spiel
- Schreibe über die Buttons den Satz "Glücksspiel kann süchtig machen!"
- gib für jeden Butten 2 Anzeigen zurück: Anzahl Versuche (wie oft wurde der Button betätigt), Anzahl Coins (wie viele Coins wurden zurückgegeben)


# Prompt
Du bist GitHub Copilot (GPT-5 mini). Lies die Datei vollständig und erstelle knapp und strukturiert in Deutsch:
1) Erzeuge eine fertige Datei bandit.py im selben Ordner mit Tkinter‑GUI und folgenden Features:
   - Drei einarmige Banditen (Slots).
   - Starten kostet 1 Coin pro Spiel.
   - Jeder Automat gibt eine zufällige Anzahl Coins zurück; die erwartete Auszahlung steigt mit der Anzahl Coins, die bereits im Automaten sind.
   - Initialisiere die Automaten mit drei unterschiedlichen Startmengen.
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
