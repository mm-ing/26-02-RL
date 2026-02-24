# RL-Workbench für cliff_walker
## Anforderungen:

- !!! WICHTIG !!!: Immer [Workbench-init.md](../Workbench-init.md) berücksichtigen.
	- Alle Regeln aus [Workbench-init.md](../Workbench-init.md) gelten verbindlich.
	- Prompt/Arbeitsweise strikt nach diesen Regeln.
    - Führe die Anweisungen in [Workbench-init.md](../Workbench-init.md) aus

## Spezielle Anforderungen
- Parameter:
	- [projektname] = `CliffWalking`
	- Nutze Environment `gymnasium.make("CliffWalking-v1")`
	- Die Environment-Visualisierung soll die Gymnasium-`CliffWalking-v1` die animierte grafische Ausgabe.
    pp  
- Algorithmen (auswählbar):
	- VDQN, DDQN
    - Nutze Stable-Baselines3
- Alle Hyperparameter der einzelnen Methoden, wie z.B. der Relay-Buffer als auch die Hyperparameter der neuronalen Netze sollen einstellbar sein. 