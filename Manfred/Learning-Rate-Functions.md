# Klassisches DQN mit explizit abnehmender Learning‑Rate
Die einfachste Form ist eine LR‑Schedule, also eine Funktion
$$\alpha _t=f(t)$$
die mit der Zeit kleiner wird.

Typische Formen:
- Exponential decay:
$$\alpha _t=\alpha _0\cdot e^{-kt}$$
- Linear decay:
$$\alpha _t=\alpha _0\cdot (1-t/T)$$
- Step decay:
$$\alpha _t=\alpha _0\cdot \gamma ^{\lfloor t/K\rfloor }$$
Diese Methode ist simpel, aber nicht besonders intelligent: sie reduziert die LR unabhängig davon, ob das Lernen stabil oder instabil ist.

## Adam / RMSProp als implizit abschwächende Learning‑Rate
Die meisten DQN‑Implementierungen nutzen Adam oder RMSProp.
Diese Optimizer haben eine adaptive Lernrate pro Parameter, die sich automatisch abschwächt, wenn:
- die Gradienten kleiner werden,
- die Updates konsistenter werden,
- die Varianz der Gradienten sinkt.
Damit entsteht eine effektiv abnehmende Learning‑Rate, ohne dass man eine explizite Funktion definieren muss.
Das ist der Grund, warum DQN fast immer mit Adam stabiler ist als mit SGD.

## Double DQN + LR‑Decay
Double DQN reduziert Overestimation und macht das Q‑Learning stabiler.
In Kombination mit einer abnehmenden LR ergibt sich:
- frühe Phase: große Schritte → schnelle Verbesserung
- späte Phase: kleine Schritte → feine Justierung
Viele Papers nutzen genau diese Kombination.

## DQN‑Varianten mit automatisch regulierter Learning‑Rate
Es gibt Methoden, die die LR nicht nur zeitabhängig, sondern fehlerabhängig anpassen.
a) Loss‑based LR Scheduling
Die LR wird reduziert, wenn der TD‑Loss nicht mehr sinkt.
Beispiel: PyTorch ReduceLROnPlateau.
b) Optimizer mit adaptiver Schrittweite
AdamW, AdaGrad, AdaDelta – alle reduzieren effektiv die Schrittweite, wenn die Parameter sich stabilisieren.
c) Natural Gradient DQN
Hier wird die LR durch die Fisher‑Information skaliert.
Das führt zu einer automatisch kleineren effektiven LR, wenn die Policy stabil wird.
d) Trust‑Region‑ähnliche DQN‑Varianten
Nicht so verbreitet wie TRPO/PPO, aber es gibt Forschungsarbeiten, die DQN‑Updates begrenzen, sodass die effektive LR sinkt, wenn die Q‑Werte zu stark schwanken.

## Distributional DQN (C51, QR‑DQN) mit LR‑Decay
Distributional Methoden sind empfindlicher gegenüber zu großen Updates.
Viele Implementierungen nutzen deshalb:
- anfangs höhere LR,
- später niedrigere LR,
- Adam als adaptive Komponente.

| Methode                     | Art der LR‑Abschwächung            | Vorteil                                  | Nachteil                                 |
|-----------------------------|--------------------------------------|-------------------------------------------|-------------------------------------------|
| LR‑Schedule (linear, exp)   | explizit, zeitabhängig              | einfach, gut kontrollierbar               | nicht adaptiv gegenüber Lernverlauf       |
| Adam / RMSProp              | implizit, gradientenabhängig        | stabil, Standard in DQN                   | schwerer theoretisch zu analysieren       |
| Loss‑based LR               | explizit, fehlerabhängig            | reagiert auf Lernfortschritt              | kann zu früh reduzieren                   |
| Natural Gradient DQN        | mathematisch skaliert               | sehr stabil                               | komplex, selten implementiert             |
| Trust‑Region‑DQN            | updateabhängig                      | verhindert Divergenz                      | experimentell, höherer Rechenaufwand      |

## Warum eine abnehmende Learning‑Rate bei DQN sinnvoll ist
- Q‑Werte konvergieren nur, wenn die Schrittweite kleiner wird
(Theorie: Robbins‑Monro‑Bedingungen).
- Späte Updates sollen feiner sein, um Overfitting und Oszillationen zu vermeiden.
- Replay‑Buffer enthält viele alte Samples → große LR führt zu Instabilität.

## Nicht‑offensichtlicher Punkt
Eine abnehmende LR wirkt besonders gut, wenn ε‑Greedy ebenfalls abnimmt.
Beide Kurven zusammen bestimmen:
- wie schnell der Agent lernt,
- wie stabil die Q‑Werte werden,
- wie gut der Agent feine Steuerimpulse lernt (z. B. beim LunarLander).

