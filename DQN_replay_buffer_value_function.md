# DQN - Zusammenhang zwischen Replay‑Buffer, Value‑Funktion und den Action‑Values des Agenten

## Der Replay‑Buffer als Datenquelle für die Value‑Funktion
Der Replay‑Buffer speichert Übergänge der Form

$$(s_t,a_t,r_t,s_{t+1},\mathrm{done}).$$

Er erfüllt drei Aufgaben:
- Er entkoppelt die Daten vom aktuellen Verhalten des Agenten (i.i.d.-ähnliche Samples).
- Er glättet die Lernverteilung (keine Katastrophen durch einzelne schlechte Episoden).
- Er liefert repräsentative Zustände, die die Value‑Funktion approximieren muss.
Ohne Replay‑Buffer würde das Q‑Netz nur auf den letzten paar Schritten lernen und sofort instabil werden.

## Die Value‑Funktion (Q‑Netz) lernt aus Replay‑Samples
Das Q‑Netz approximiert die Funktion

$$Q_{\theta }(s,a)\approx \mathbb{E}[R\mid s,a].$$

Für jedes Sample aus dem Replay‑Buffer wird ein TD‑Target berechnet:

$$y=r+\gamma \max _{a'}Q_{\theta ^-}(s_{t+1},a').$$

Dabei:

$$Q_{\theta }$$ = aktuelles Netz (wird trainiert)

$$Q_{\theta ^-}$$ = Target‑Netz (stabilisiert das Lernen)

Der Replay‑Buffer bestimmt also welche Verteilungen von Zuständen und Aktionen die Value‑Funktion überhaupt sieht.
Die Value‑Funktion bestimmt wie gut diese Samples erklärt werden.

## Die Action‑Values steuern das Verhalten des Agenten
Der Agent wählt Aktionen über:

$$a_t=\arg \max _aQ_{\theta }(s_t,a)$$

mit ε‑Greedy‑Exploration.
Damit beeinflusst die Value‑Funktion direkt:
- welche Aktionen ausprobiert werden,
- welche Trajektorien entstehen,
- welche Daten in den Replay‑Buffer gelangen.
Das Verhalten des Agenten bestimmt also den Input des Replay‑Buffers.

Der geschlossene Kreislauf: Wie alles zusammenwirkt
1. Agent handelt → erzeugt Daten
Die aktuellen Action‑Values bestimmen die Aktion.
Diese Aktion erzeugt einen Übergang, der in den Replay‑Buffer wandert.
2. Replay‑Buffer liefert Trainingsdaten
Das Q‑Netz trainiert auf zufälligen Samples aus dem Buffer.
Die Qualität und Vielfalt dieser Samples bestimmt die Lernrichtung.
3. Q‑Netz verbessert seine Value‑Schätzung
Durch TD‑Updates wird die Value‑Funktion korrigiert.
Sie wird besser darin, zukünftige Rewards vorherzusagen.
4. Neue Action‑Values → neues Verhalten
Das verbesserte Q‑Netz führt zu besseren Aktionen.
Diese erzeugen bessere Daten.
Diese verbessern das Q‑Netz weiter.
Das ist der DQN‑Lernkreislauf.

Warum dieser Zusammenhang so empfindlich ist
1. Replay‑Buffer beeinflusst die Stabilität der Value‑Funktion
- Zu kleiner Buffer → Korrelationen → Divergenz
- Zu großer Buffer → veraltete Daten → langsame Anpassung
2. Value‑Funktion beeinflusst die Qualität der Daten
Wenn das Q‑Netz früh falsche Werte lernt:
- wählt der Agent schlechte Aktionen,
- erzeugt schlechte Trajektorien,
- die wiederum das Q‑Netz weiter verschlechtern.
3. Action‑Values bestimmen die Exploration
Wenn ε zu schnell sinkt:
- der Buffer enthält zu wenig erfolgreiche Landungen,
- das Q‑Netz lernt nie, wie man landet.

| Komponente        | Rolle                               | Einfluss auf andere                                      |
|-------------------|--------------------------------------|-----------------------------------------------------------|
| Replay‑Buffer     | Speichert Übergänge                  | Bestimmt die Datenverteilung für Q‑Updates               |
| Value‑Funktion    | Approximiert `Q(s,a)`                | Bestimmt die Action‑Values und damit das Verhalten       |
| Action‑Values     | Steuern die Policy                   | Bestimmen, welche Daten in den Replay‑Buffer gelangen    |

Ein nicht‑offensichtlicher, aber zentraler Punkt
Der Replay‑Buffer ist die einzige Verbindung zwischen Verhalten und Lernen.
Er ist das Gedächtnis des Agenten.
Die Value‑Funktion ist die Interpretation dieses Gedächtnisses.
Die Action‑Values sind die Konsequenz dieser Interpretation.
Wenn eines der drei Elemente instabil ist, bricht der gesamte Kreislauf zusammen.
