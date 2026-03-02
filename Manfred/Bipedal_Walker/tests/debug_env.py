"""Debug SB3 and gymnasium episode tracking."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Test 1: Check max_episode_steps
print("=== Test 1: max_episode_steps ===")
env = gym.make("BipedalWalker-v3", max_episode_steps=50)
obs, _ = env.reset()
done = False
steps = 0
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    steps += 1
    done = term or trunc
print(f"Episode ended after {steps} steps (expected ~50), term={term}, trunc={trunc}")
env.close()

# Test 2: DummyVecEnv with max_episode_steps
print("\n=== Test 2: DummyVecEnv with max_episode_steps ===")
def make_env():
    return gym.make("BipedalWalker-v3", max_episode_steps=50)

vec_env = DummyVecEnv([make_env])
obs = vec_env.reset()
ep_count = 0
for i in range(300):
    obs, r, dones, infos = vec_env.step(vec_env.action_space.sample())
    if dones[0]:
        ep_count += 1
        print(f"  VecEnv episode {ep_count} done at step {i}")
    for info in infos:
        if "episode" in info:
            print(f"  Episode info at step {i}: r={info['episode']['r']:.2f}, l={info['episode']['l']}")
print(f"Total episodes with dones: {ep_count}")
vec_env.close()

# Test 3: PPO with callback
print("\n=== Test 3: PPO callback with infos ===")
class DebugCB(BaseCallback):
    def __init__(self):
        super().__init__()
        self.eps = 0
        self.steps = 0
    def _on_step(self):
        self.steps += 1
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if any(dones):
            print(f"  dones at step {self.steps}: {dones}, infos: {infos}")
        for info in infos:
            if "episode" in info:
                self.eps += 1
                print(f"  EP DONE: r={info['episode']['r']:.2f}, l={info['episode']['l']}")
        return True
    def _on_rollout_end(self):
        print(f"  Rollout end, steps so far: {self.steps}")

vec_env2 = DummyVecEnv([make_env])
model = PPO("MlpPolicy", vec_env2, n_steps=64, batch_size=32, verbose=0)
cb = DebugCB()
model.learn(total_timesteps=300, callback=cb)
print(f"PPO episodes tracked: {cb.eps}, step calls: {cb.steps}")
