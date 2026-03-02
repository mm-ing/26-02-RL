import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class DebugCB(BaseCallback):
    def __init__(self):
        super().__init__()
        self.eps = 0
        self.step_count = 0
    def _on_step(self):
        self.step_count += 1
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.eps += 1
                ep = info['episode']
                print(f"Episode {self.eps}: r={ep['r']:.2f}, l={ep['l']}")
        return True

env = DummyVecEnv([lambda: gym.make('BipedalWalker-v3')])
model = PPO('MlpPolicy', env, n_steps=64, batch_size=32, verbose=0)
cb = DebugCB()
model.learn(total_timesteps=500, callback=cb)
print(f'Total episodes: {cb.eps}, steps called: {cb.step_count}')
