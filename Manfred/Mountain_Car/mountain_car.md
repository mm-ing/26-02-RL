# RL-Workbench reinforcement learning für gymnasium cartpole

## Vorrausetzungen:
- !!! WICHTIG !!!: Immer [Workbench.md](../Workbench.md) berücksichtigen.
	- Alle Regeln aus [Workbench.md](../Workbench.md) gelten verbindlich.
	- Prompt/Arbeitsweise strikt nach diesen Regeln.
    - Führe die Anweisungen in [Workbench.md](../Workbench.md) aus

## Spezielle Anforderungen

### Projekt
- [projektname] = `mountain_car`

### Animation
- Nutze Environment `gymnasium.make("MountainCar-v0")`
- Die Environment-Visualisierung soll die Gymnasium-`MountainCar-v0` die animierte grafische Ausgabe.
- Parameter
    - sutton_barto_reward (default True)
     
### Algorithmen (auswählbar):
- D3QN (Double + Dueling)
- Double DQN + Prioritized Experience Replay
- Dueling DQN
- Nutze Stable-Baselines3
- Alle Hyperparameter der einzelnen Methoden, wie z.B. der Relay-Buffer als auch die Hyperparameter der neuronalen Netze sollen einstellbar sein. 

---




