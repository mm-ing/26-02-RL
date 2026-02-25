"""Unit + simulation tests for Frozen Lake RL workbench."""

import os
import sys
import tempfile
import threading
import time

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frozen_lake_logic import (
    AlgorithmConfig,
    CheckpointManager,
    DoubleDQN,
    EnvironmentRegistry,
    EpisodeResult,
    Event,
    EventBus,
    EventType,
    JobStatus,
    NetworkConfig,
    OneHotWrapper,
    TrainingJob,
    TrainingManager,
    build_model,
    make_env,
)


def quick_config(**overrides) -> AlgorithmConfig:
    cfg = AlgorithmConfig(
        episodes=6,
        max_steps=30,
        learning_starts=8,
        buffer_size=256,
        batch_size=16,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=20,
        exploration_fraction=0.4,
        learning_rate=8e-4,
        network=NetworkConfig(hidden_layers=[32, 32], activation="ReLU"),
        env_name="FrozenLake-v1",
        is_slippery=False,
        map_name="4x4",
        success_rate=1.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestEnvironment:
    def test_make_env(self):
        env = make_env(is_slippery=False, map_name="4x4", success_rate=1.0)
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert int(obs.sum()) == 1
        env.close()

    def test_render(self):
        env = make_env(render_mode="rgb_array", is_slippery=False)
        env.reset()
        frame = env.render()
        assert frame is not None
        assert frame.ndim == 3
        env.close()

    def test_registry(self):
        reg = EnvironmentRegistry()
        reg.register("frozen", lambda **kw: __import__("frozen_lake_logic").Environment("FrozenLake-v1", **kw))
        env = reg.create("frozen", is_slippery=False, map_name="4x4", success_rate=1.0)
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()


class TestEvents:
    def test_publish_subscribe(self):
        bus = EventBus()
        got = []
        bus.subscribe(lambda e: got.append(e))
        bus.publish(Event(EventType.JOB_CREATED, {"job_id": "abc"}))
        bus.process_events()
        assert len(got) == 1

    def test_thread_safety(self):
        bus = EventBus()
        counter = {"n": 0}
        bus.subscribe(lambda _e: counter.__setitem__("n", counter["n"] + 1))

        def push():
            for _ in range(40):
                bus.publish(Event(EventType.STEP_COMPLETED, {}))

        threads = [threading.Thread(target=push) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        bus.process_events(max_events=500)
        assert counter["n"] == 160


class TestModel:
    @pytest.mark.parametrize("algo", ["VDQN", "DDQN", "Dueling DQN", "Prioritized DQN"])
    def test_build(self, algo):
        env = make_env(is_slippery=False, map_name="4x4")
        cfg = quick_config(algorithm=algo)
        model = build_model(cfg, env)
        assert model is not None
        if algo == "DDQN":
            assert isinstance(model, DoubleDQN)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert 0 <= int(action) < int(env.action_space.n)
        env.close()


class TestJobs:
    def test_record(self):
        job = TrainingJob(quick_config(), name="job")
        job.record_episode(EpisodeResult(episode=1, total_reward=1.0, steps=3, duration=0.1))
        assert job.total_episodes_done == 1

    def test_manager_modes(self):
        bus = EventBus()
        mgr = TrainingManager(bus)
        jobs = mgr.add_compare_jobs(quick_config())
        assert len(jobs) == 4
        tuned = mgr.add_tuning_jobs(quick_config(), "learning_rate", 0.001, 0.005, 0.002)
        assert len(tuned) == 3

    def test_save_load(self):
        job = TrainingJob(quick_config(), name="save_job")
        job.episode_returns = [0.0, 1.0]
        job.episode_lengths = [4, 4]
        with tempfile.TemporaryDirectory() as td:
            CheckpointManager.save_job(job, td)
            loaded = CheckpointManager.load_job(td)
            assert loaded.name == "save_job"
            assert loaded.episode_returns == [0.0, 1.0]


class TinyLearnEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.render_mode = render_mode
        self.state = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        reward = 1.0 if int(action) == 1 else 0.0
        self.state = 1
        return self.state, reward, True, False, {}

    def render(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)


def run_learning_smoke(algo: str):
    import frozen_lake_logic as logic

    original_make_env = logic.make_env

    def fake_make_env(*args, **kwargs):
        env = TinyLearnEnv(render_mode=kwargs.get("render_mode"))
        env = OneHotWrapper(env)
        return gym.wrappers.RecordEpisodeStatistics(env)

    logic.make_env = fake_make_env
    try:
        bus = EventBus()
        mgr = TrainingManager(bus)
        cfg = quick_config(algorithm=algo, episodes=20, max_steps=1, learning_starts=4, buffer_size=64, batch_size=8)
        job = mgr.add_job(cfg, name=f"sim_{algo}")
        results = []
        bus.subscribe(lambda e: results.append(e.data["result"].total_reward)
                      if e.type == EventType.EPISODE_COMPLETED and e.data.get("job_id") == job.job_id else None)
        job.start_training(bus, additional_episodes=cfg.episodes)
        for _ in range(250):
            bus.process_events()
            if not job.is_alive():
                break
            time.sleep(0.02)
        bus.process_events()
        assert len(results) >= 5
        assert np.mean(results[-5:]) >= 0.4
    finally:
        logic.make_env = original_make_env


@pytest.mark.parametrize("algo", ["VDQN", "DDQN", "Dueling DQN", "Prioritized DQN"])
def test_algorithm_learning(algo):
    run_learning_smoke(algo)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
