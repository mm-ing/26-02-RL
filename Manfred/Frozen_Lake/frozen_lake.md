# RL-Workbench für frozen-lake

## Vorrausetzungen:
- !!! WICHTIG !!!: Immer [Workbench.md](../Workbench.md) berücksichtigen.
	- Alle Regeln aus [Workbench.md](../Workbench.md) gelten verbindlich.
	- Prompt/Arbeitsweise strikt nach diesen Regeln.
    - Führe die Anweisungen in [Workbench.md](../Workbench.md) aus

## Spezielle Anforderungen

### Projekt
- [projektname] = `frozen_lake`

### Animation
- Nutze Environment `gymnasium.make("FrozenLake-v1")`
- Die Environment-Visualisierung soll die Gymnasium-`FrozenLake-v1` die animierte grafische Ausgabe.
- Parameter
    - is_slippery (default false)
    - map_name: 4x4 / 8x8 (default 4x4)
    - success_rate 1.0 / 3.0
     
### Algorithmen (auswählbar):
- VDQN (vanilla dqn)
- DDQN (double dqn)
- Dueling DQN
- prioritized DQN
- Nutze Stable-Baselines3
- Alle Hyperparameter der einzelnen Methoden, wie z.B. der Relay-Buffer als auch die Hyperparameter der neuronalen Netze sollen einstellbar sein. 