# RL-Workbench reinforcement learning für gymnasium taxi

## Vorrausetzungen:
- !!! WICHTIG !!!: Immer [Workbench.md](../Workbench.md) berücksichtigen.
	- Alle Regeln aus [Workbench.md](../Workbench.md) gelten verbindlich.
	- Prompt/Arbeitsweise strikt nach diesen Regeln.
    - Führe die Anweisungen in [Workbench.md](../Workbench.md) aus

## Spezielle Anforderungen

### Projekt
- [projektname] = `taxi`

### Animation
- Nutze Environment `gymnasium.make("Taxi-v3")`
- Die Environment-Visualisierung soll die Gymnasium-`Taxi-v3` die animierte grafische Ausgabe.
- Parameter
    - is_raining (default false)
    - fickle_passenger (default false)
     
### Algorithmen (auswählbar):
- VDQN (vanilla dqn)
- DDQN (double dqn)
- Dueling DQN
- prioritized DQN
- Nutze Stable-Baselines3
- Alle Hyperparameter der einzelnen Methoden, wie z.B. der Relay-Buffer als auch die Hyperparameter der neuronalen Netze sollen einstellbar sein. 