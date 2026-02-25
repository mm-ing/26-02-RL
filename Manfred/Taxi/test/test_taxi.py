"""Unit + simulation tests for Taxi RL workbench."""

import os
import sys
import tempfile
import threading
import time

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from taxi_logic import (
    AlgorithmConfig,
    CheckpointManager,
    DoubleDQN,
    Environment,
    EnvironmentRegistry,
    EpisodeResult,
    Event,
    EventBus,
    EventType,
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
        max_steps=60,
        learning_starts=8,
        buffer_size=256,
        batch_size=16,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=20,
        exploration_fraction=0.4,
        learning_rate=8e-4,
        network=NetworkConfig(hidden_layers=[64, 64], activation="ReLU"),
        env_name="Taxi-v3",
        is_raining=False,
        fickle_passenger=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class TestEnvironment:
    def test_make_env(self):
        env = make_env(is_raining=False, fickle_passenger=False)
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape[0] == 500
        assert int(obs.sum()) == 1
        env.close()

    def test_render(self):
        env = make_env(render_mode="rgb_array", is_raining=False, fickle_passenger=False)
        env.reset()
        frame = np.asarray(env.render())
        assert frame is not None
        assert frame.ndim == 3
        env.close()

    def test_registry(self):
        registry = EnvironmentRegistry()
        registry.register("taxi-custom", lambda **kw: Environment(env_name="Taxi-v3", **kw))
        env = registry.create("taxi-custom", is_raining=False, fickle_passenger=False)
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()


class TestEvents:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda event: received.append(event))
        bus.publish(Event(EventType.JOB_CREATED, {"job_id": "abc"}))
        bus.process_events()
        assert len(received) == 1

    def test_thread_safety(self):
        bus = EventBus()
        counter = {"n": 0}
        bus.subscribe(lambda _event: counter.__setitem__("n", counter["n"] + 1))

        def producer():
            for _ in range(40):
                bus.publish(Event(EventType.STEP_COMPLETED, {}))

        threads = [threading.Thread(target=producer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        bus.process_events(max_events=500)
        assert counter["n"] == 160


class TestModel:
    @pytest.mark.parametrize("algo", ["VDQN", "DDQN", "Dueling DQN", "Prioritized DQN"])
    def test_build(self, algo):
        env = make_env(is_raining=False, fickle_passenger=False)
        cfg = quick_config(algorithm=algo)
        model = build_model(cfg, env)
        assert model is not None
        if algo == "DDQN":
            assert isinstance(model, DoubleDQN)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        action_space_n = int(getattr(env.action_space, "n", 6))
        assert 0 <= int(action) < action_space_n
        env.close()


class TestJobs:
    def test_record(self):
        job = TrainingJob(quick_config(), name="job")
        job.record_episode(EpisodeResult(episode=1, total_reward=1.0, steps=3, duration=0.1))
        assert job.total_episodes_done == 1

    def test_manager_modes(self):
        bus = EventBus()
        manager = TrainingManager(bus)
        jobs = manager.add_compare_jobs(quick_config())
        assert len(jobs) == 4
        tuned = manager.add_tuning_jobs(quick_config(), "learning_rate", 0.001, 0.005, 0.002)
        assert len(tuned) == 3

    def test_save_load(self):
        job = TrainingJob(quick_config(), name="save_job")
        job.episode_returns = [0.0, 1.0]
        job.episode_lengths = [4, 4]
        with tempfile.TemporaryDirectory() as tmpdir:
            CheckpointManager.save_job(job, tmpdir)
            loaded = CheckpointManager.load_job(tmpdir)
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
    import taxi_logic as logic

    original_make_env = logic.make_env

    def fake_make_env(*args, **kwargs):
        env = TinyLearnEnv(render_mode=kwargs.get("render_mode"))
        env = OneHotWrapper(env)
        return gym.wrappers.RecordEpisodeStatistics(env)

    logic.make_env = fake_make_env
    try:
        bus = EventBus()
        manager = TrainingManager(bus)
        cfg = quick_config(algorithm=algo, episodes=20, max_steps=1, learning_starts=4, buffer_size=64, batch_size=8)
        job = manager.add_job(cfg, name=f"sim_{algo}")
        returns = []
        bus.subscribe(
            lambda event: returns.append(event.data["result"].total_reward)
            if event.type == EventType.EPISODE_COMPLETED and event.data.get("job_id") == job.job_id
            else None
        )
        job.start_training(bus, additional_episodes=cfg.episodes)
        for _ in range(250):
            bus.process_events()
            if not job.is_alive():
                break
            time.sleep(0.02)
        bus.process_events()
        assert len(returns) >= 5
        assert np.mean(returns[-5:]) >= 0.4
    finally:
        logic.make_env = original_make_env


@pytest.mark.parametrize("algo", ["VDQN", "DDQN", "Dueling DQN", "Prioritized DQN"])
def test_algorithm_learning(algo):
    run_learning_smoke(algo)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
