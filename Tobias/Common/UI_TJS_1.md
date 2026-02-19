Bitte erstelle mir aus folgenden Anweisungen einen Prompt, den Du gut verstehst:
Erstelle ein Python 3-Programm mit tkinter, das eine GUI mit drei Banditen (Buttons) bereitstellt.
Bennene diese drei Buttons als Bandit mit laufender Nummer.
Füge oben über alle drei Banditen "Viel Erfolg Dir!!!" ein.
Jedem Banditen wird beim Programmstart eine Auszahlungswahrscheinlichkeit fest zugewiesen.
Bandit 1: 20%. Bandit 2: 40%. Bandit 3: 80%.
Nur mit dieser Wahrscheinlichkeit soll bei jedem Klick eines Banditen berechnet werden, ob überhaupt etwas ausgezahlt wird.
Die mögliche Auszahlung soll also Bernoulli-Logik entsprechen.
Jeder Bandit hat zwei Labels daneben/unterhalb:
"Anzahl Klicks: N" — zählt, wie oft der Bandit angeklickt wurde.
"Ausgabewert: X" — zeigt den zuletzt erzeugten Ausgabewert und die Summe der bisherigen Auszahlungen.
Klick-Verhalten: Wenn ein Bandit angeklickt wird, wird die Klick-Anzahl erhöht und ein neuer Ausgabewert generiert:
Erstelle je Bandit zwei Labels, die Anzahl Klicks und Ausgabewert anzeigen.
Der Nutzer soll die drei Banditen anklicken können und damit aktivieren.
Nach jedem Klick die beiden Labels des jeweiligen Banditen aktualisieren.
UI muss responsiv sein und wiederholte Klicks erlauben.
Zeige die Wahrscheinlichkeit p als zusätzliches kleines Label pro Bandit.
Gib sauberen, gut lesbaren Python-Code aus, der sofort lauffähig ist (inkl. benötigter Importe).





Fenster mit drei Buttons: "Bandit 1", "Bandit 2", "Bandit 3".
- Jedem Banditen wird beim Programmstart eine Auszahlungswahrscheinlichkeit zugewiesen.
Bandit 1: 20%. Bandit 2: 40%. Bandit 3: 80%.
- Jeder Bandit hat zwei Labels daneben/unterhalb:
  1) "Anzahl Klicks: N" — zählt, wie oft der Bandit angeklickt wurde.
  2) "Ausgabewert: X" — zeigt den zuletzt erzeugten Ausgabewert und die Summe der bisherigen Auszahlungen.
- Klick-Verhalten: Wenn ein Bandit angeklickt wird, wird die Klick-Anzahl erhöht und ein neuer Ausgabewert generiert:
  - Mit Wahrscheinlichkeit p wird ein zufälliger Integer gleichverteilt aus [50, 100] erzeugt.
  - Mit Wahrscheinlichkeit 1-p wird ein zufälliger Integer gleichverteilt aus [0, 49] erzeugt.
- Nach jedem Klick die beiden Labels des jeweiligen Banditen aktualisieren.
- UI muss responsiv sein und wiederholte Klicks erlauben.
- Optional: Zeige die Wahrscheinlichkeit p als zusätzliches kleines Label pro Bandit.




Erstelle ein Python 3-Programm mit tkinter, das eine GUI mit drei Banditen (Buttons) bereitstellt.

Anforderungen:
- Fenster mit drei Buttons: "Bandit 1", "Bandit 2", "Bandit 3".
- Jedem Banditen wird beim Programmstart eine Auszahlungswahrscheinlichkeit zugewiesen.
Bandit 1: 20%. Bandit 2: 40%. Bandit 3: 80%.
- Jeder Bandit hat zwei Labels daneben/unterhalb:
  1) "Anzahl Klicks: N" — zählt, wie oft der Bandit angeklickt wurde.
  2) "Ausgabewert: X" — zeigt den zuletzt erzeugten Ausgabewert und die Summe der bisherigen Auszahlungen.
- Klick-Verhalten: Wenn ein Bandit angeklickt wird, wird die Klick-Anzahl erhöht und ein neuer Ausgabewert generiert:
  - Mit Wahrscheinlichkeit p wird ein zufälliger Integer gleichverteilt aus [50, 100] erzeugt.
  - Mit Wahrscheinlichkeit 1-p wird ein zufälliger Integer gleichverteilt aus [0, 49] erzeugt.
- Nach jedem Klick die beiden Labels des jeweiligen Banditen aktualisieren.
- UI muss responsiv sein und wiederholte Klicks erlauben.
- Optional: Zeige die Wahrscheinlichkeit p als zusätzliches kleines Label pro Bandit.

Gib sauberen, gut lesbaren Python-Code aus, der sofort lauffähig ist (inkl. benötigter Importe).
Speichere dies unter UI_TJS_2.py!



Erstelle ein Python 3‑Programm (Datei: UI_TJS_2.py) mit tkinter, das eine GUI für drei Banditen bereitstellt.

Anforderungen:
- Fenster oben zentriert: "Viel Erfolg Dir!!!".
- Drei Buttons beschriftet "Bandit 1", "Bandit 2", "Bandit 3".
- Feste Auszahlungswahrscheinlichkeiten beim Start:
  - Bandit 1: p = 0.20
  - Bandit 2: p = 0.40
  - Bandit 3: p = 0.80
- Pro Bandit ein kleines Label mit der Wahrscheinlichkeit (z. B. "p = 20%").
- Je Bandit zwei Status‑Labels:
  1) "Anzahl Klicks: N" — zählt die Klicks.
  2) "Ausgabewert: L (Summe: S)" — zeigt zuletzt erzeugten Integer L und kumulierte Summe S.
- Klick‑Verhalten:
  - Bei Klick erhöhe Klick‑Zähler.
  - Führe eine Bernoulli‑Probe mit p durch:
    - Bei Erfolg: sample uniformen Integer aus [50,100].
    - Bei Misserfolg: sample uniformen Integer aus [0,49].
  - Aktualisiere nach jedem Klick sofort die beiden Labels des betroffenen Banditen.
- UI muss responsiv sein und wiederholte Klicks ohne Neustart erlauben.
- Nur Standardbibliothek verwenden; vollständiger, sauberer und sofort lauffähiger Python‑Code (inkl. benötigter Importe) zurückgeben, geeignet zum Speichern als UI_TJS_2.py.
```Erstelle ein Python 3‑Programm (Datei: UI_TJS_2.py) mit tkinter, das eine GUI für drei Banditen bereitstellt.

Anforderungen:
- Fenster oben zentriert: "Viel Erfolg Dir!!!".
- Drei Buttons beschriftet "Bandit 1", "Bandit 2", "Bandit 3".
- Feste Auszahlungswahrscheinlichkeiten beim Start:
  - Bandit 1: p = 0.20
  - Bandit 2: p = 0.40
  - Bandit 3: p = 0.80
- Pro Bandit ein kleines Label mit der Wahrscheinlichkeit (z. B. "p = 20%").
- Je Bandit zwei Status‑Labels:
  1) "Anzahl Klicks: N" — zählt die Klicks.
  2) "Ausgabewert: L (Summe: S)" — zeigt zuletzt erzeugten Integer L und kumulierte Summe S.
- Klick‑Verhalten:
  - Bei Klick erhöhe Klick‑Zähler.
  - Führe eine Bernoulli‑Probe mit p durch:
    - Bei Erfolg: sample uniformen Integer aus [50,100].
    - Bei Misserfolg: sample uniformen Integer aus [0,49].
  - Aktualisiere nach jedem Klick sofort die beiden Labels des betroffenen Banditen.
- UI muss responsiv sein und wiederholte Klicks ohne Neustart erlauben.
- Nur Standardbibliothek verwenden; vollständiger, sauberer und sofort lauffähiger Python‑Code (inkl. benötigter Importe) zurückgeben, geeignet zum Speichern als UI_TJS_2.py.