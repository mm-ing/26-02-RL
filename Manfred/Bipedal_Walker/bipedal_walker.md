# RL-Workbench reinforcement learning für gymnasium Bipedal-Walker

## Vorrausetzungen:
- !!! WICHTIG !!!: Immer [Workbench.md](../Workbench.md) berücksichtigen.
	- Alle Regeln aus [Workbench.md](../Workbench.md) gelten verbindlich.
	- Prompt/Arbeitsweise strikt nach diesen Regeln.
    - Führe die Anweisungen in [Workbench.md](../Workbench.md) aus

## Spezielle Anforderungen

### Projekt
- [projektname] = `bipedal_walker`

### Animation
- Nutze Environment `gymnasium.make("BipedalWalker-v3")`
- Die Environment-Visualisierung soll die Gymnasium-`BipedalWalker-v3` die animierte grafische Ausgabe.
- Parameter
    - hardcore (default False)
     
### Algorithmen (auswählbar):
- PPO
- A2C
- SAC
- TD3
- Nutze Stable-Baselines3
- Alle Hyperparameter der einzelnen Methoden, wie z.B. der Relay-Buffer als auch die Hyperparameter der neuronalen Netze sollen einstellbar sein. 

---