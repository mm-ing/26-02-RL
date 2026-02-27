# RL-Workbench reinforcement learning für gymnasium LunarLander

## Vorrausetzungen:
- !!! WICHTIG !!!: Immer [Workbench.md](../Workbench.md) berücksichtigen.
	- Alle Regeln aus [Workbench.md](../Workbench.md) gelten verbindlich.
	- Prompt/Arbeitsweise strikt nach diesen Regeln.
    - Führe die Anweisungen in [Workbench.md](../Workbench.md) aus

## Spezielle Anforderungen

### Projekt
- [projektname] = `lunar_lander`

### Animation
- Nutze Environment `gymnasium.make("LunarLander-v3")`
- Die Environment-Visualisierung soll die Gymnasium-`LunarLander-v3` die animierte grafische Ausgabe.
- Parameter
    - continuous (default False)
    - gravity (default -10.0)
    - enable_wind (default False)
    - wind_power (default 15.0)
    - turbulence_power (default 1.5)
     
### Algorithmen (auswählbar):
- D3QN (Double + Dueling)
- Double DQN + Prioritized Experience Replay
- Dueling DQN
- Nutze Stable-Baselines3
- Alle Hyperparameter der einzelnen Methoden, wie z.B. der Relay-Buffer als auch die Hyperparameter der neuronalen Netze sollen einstellbar sein. 

---