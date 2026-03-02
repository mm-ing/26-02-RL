"""Deep debug of SB3 callback locals."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return gym.make("BipedalWalker-v3", max_episode_steps=50)

class DeepDebugCB(BaseCallback):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.ep_count = 0
        self.printed_locals = False

    def _on_step(self):
        self.call_count += 1

        # Print locals on first call
        if not self.printed_locals:
            print(f"\n[Call 1] locals keys: {sorted(self.locals.keys())}")
            if "dones" in self.locals:
                print(f"  dones type: {type(self.locals['dones'])}, val: {self.locals['dones']}")
            if "infos" in self.locals:
                print(f"  infos type: {type(self.locals['infos'])}, val: {self.locals['infos']}")
            self.printed_locals = True

        # Check for dones
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        if dones is not None and len(dones) > 0 and any(dones):
            print(f"  [Step {self.call_count}] DONE! dones={dones}")
            print(f"    infos={infos}")
            for info in infos:
                if "episode" in info:
                    self.ep_count += 1
                    print(f"    Episode {self.ep_count}: r={info['episode']['r']:.2f}")

        return True

    def _on_rollout_end(self):
        print(f"[Rollout end at step {self.call_count}], ep_count={self.ep_count}")


vec_env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, verbose=0)
print(f"Env wrapped with: {type(model.env)}")

cb = DeepDebugCB()
model.learn(total_timesteps=300, callback=cb)
print(f"\nFinal: step calls={cb.call_count}, episodes={cb.ep_count}")
