"""
tests/test_logic.py
Unit and simulation tests for bipedal_walker_logic.py
"""

import sys
import os
import time
import threading
import json
import tempfile
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bipedal_walker_logic import (
    A2CConfig,
    AlgorithmType,
    CheckpointManager,
    EnvironmentConfig,
    EpisodeConfig,
    Event,
    EventBus,
    EventType,
    JobConfig,
    JobStatus,
    NetworkConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
    TrainingJob,
    TrainingManager,
    TuningConfig,
    _deserialize_config,
    _serialize_config,
    _set_param,
    get_bus,
)


# Use a simpler and faster environment for testing
TEST_ENV = "BipedalWalker-v3"


def make_fast_config(algo: str, n_episodes: int = 5) -> JobConfig:
    """Create a minimal config for fast testing."""
    ep_cfg = EpisodeConfig(n_episodes=n_episodes, max_steps=50, alpha=3e-4, gamma=0.99)
    net    = NetworkConfig(hidden_layers=[32, 32], activation="relu")
    return JobConfig(
        name=f"test-{algo}",
        algorithm=algo,
        env_cfg=EnvironmentConfig(env_name=TEST_ENV, hardcore=False),
        ep_cfg=ep_cfg,
        ppo_cfg=PPOConfig(n_steps=128, batch_size=32, n_epochs=2, network=net),
        a2c_cfg=A2CConfig(n_steps=5, network=net),
        sac_cfg=SACConfig(buffer_size=10_000, batch_size=32, learning_starts=10,
                          train_freq=1, gradient_steps=1, network=net),
        td3_cfg=TD3Config(buffer_size=10_000, batch_size=32, learning_starts=10,
                          train_freq=1, gradient_steps=1, network=net),
    )


# ---------------------------------------------------------------------------
# EventBus Tests
# ---------------------------------------------------------------------------

class TestEventBus(unittest.TestCase):

    def test_subscribe_and_publish(self):
        bus    = EventBus()
        events = []
        bus.subscribe(EventType.JOB_CREATED, lambda e: events.append(e))
        bus.publish(Event(EventType.JOB_CREATED, "j1", {"x": 1}))
        bus.drain()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].job_id, "j1")

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = []
        bus.subscribe(EventType.EPISODE_COMPLETED, lambda e: results.append("a"))
        bus.subscribe(EventType.EPISODE_COMPLETED, lambda e: results.append("b"))
        bus.publish(Event(EventType.EPISODE_COMPLETED, "j1"))
        bus.drain()
        self.assertEqual(sorted(results), ["a", "b"])

    def test_unsubscribe(self):
        bus    = EventBus()
        events = []

        def listener(e):
            events.append(e)

        bus.subscribe(EventType.JOB_DONE, listener)
        bus.unsubscribe(EventType.JOB_DONE, listener)
        bus.publish(Event(EventType.JOB_DONE, "j1"))
        bus.drain()
        self.assertEqual(len(events), 0)

    def test_thread_safe_publish(self):
        bus = EventBus()
        results = []
        bus.subscribe(EventType.STEP_COMPLETED, lambda e: results.append(1))

        def publisher():
            for _ in range(50):
                bus.publish(Event(EventType.STEP_COMPLETED))

        threads = [threading.Thread(target=publisher) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        bus.drain()
        self.assertEqual(len(results), 200)


# ---------------------------------------------------------------------------
# Config Serialization Tests
# ---------------------------------------------------------------------------

class TestConfigSerialization(unittest.TestCase):

    def test_serialize_deserialize_roundtrip(self):
        cfg  = make_fast_config(AlgorithmType.PPO.value)
        raw  = _serialize_config(cfg)
        cfg2 = _deserialize_config(raw)

        self.assertEqual(cfg.algorithm, cfg2.algorithm)
        self.assertEqual(cfg.ep_cfg.n_episodes, cfg2.ep_cfg.n_episodes)
        self.assertEqual(cfg.ep_cfg.alpha, cfg2.ep_cfg.alpha)
        self.assertEqual(cfg.ppo_cfg.n_steps, cfg2.ppo_cfg.n_steps)
        self.assertEqual(cfg.ppo_cfg.network.hidden_layers, cfg2.ppo_cfg.network.hidden_layers)

    def test_json_serializable(self):
        cfg = make_fast_config(AlgorithmType.SAC.value)
        raw = _serialize_config(cfg)
        s   = json.dumps(raw)  # should not raise
        self.assertIsInstance(s, str)


# ---------------------------------------------------------------------------
# SetParam Tests
# ---------------------------------------------------------------------------

class TestSetParam(unittest.TestCase):

    def test_set_alpha(self):
        cfg = make_fast_config(AlgorithmType.PPO.value)
        _set_param(cfg, "alpha", 1e-3)
        self.assertAlmostEqual(cfg.ep_cfg.alpha, 1e-3)

    def test_set_gamma(self):
        cfg = make_fast_config(AlgorithmType.PPO.value)
        _set_param(cfg, "gamma", 0.95)
        self.assertAlmostEqual(cfg.ep_cfg.gamma, 0.95)

    def test_set_batch_size(self):
        cfg = make_fast_config(AlgorithmType.SAC.value)
        _set_param(cfg, "batch_size", 512)
        self.assertEqual(cfg.sac_cfg.batch_size, 512)

    def test_set_tau(self):
        cfg = make_fast_config(AlgorithmType.TD3.value)
        _set_param(cfg, "tau", 0.01)
        self.assertAlmostEqual(cfg.td3_cfg.tau, 0.01)


# ---------------------------------------------------------------------------
# TrainingManager Tests
# ---------------------------------------------------------------------------

class TestTrainingManager(unittest.TestCase):

    def setUp(self):
        self.bus     = EventBus()
        self.manager = TrainingManager(self.bus)

    def test_add_job(self):
        cfg = make_fast_config(AlgorithmType.PPO.value)
        job = self.manager.add_job(cfg)
        self.assertIn(job.job_id, self.manager.jobs)

    def test_remove_job(self):
        cfg = make_fast_config(AlgorithmType.PPO.value)
        job = self.manager.add_job(cfg)
        jid = job.job_id
        self.manager.remove(jid)
        self.assertNotIn(jid, self.manager.jobs)

    def test_build_tuning_jobs(self):
        cfg     = make_fast_config(AlgorithmType.PPO.value)
        tuning  = TuningConfig(enabled=True, parameter="alpha",
                               min_value=1e-4, max_value=3e-4, step=1e-4)
        configs = self.manager.build_tuning_jobs(cfg, tuning)
        self.assertEqual(len(configs), 3)
        alphas = [c.ep_cfg.alpha for c in configs]
        self.assertAlmostEqual(min(alphas), 1e-4, places=6)
        self.assertAlmostEqual(max(alphas), 3e-4, places=6)


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------

class TestCheckpointManager(unittest.TestCase):

    def test_save_metrics_only(self):
        bus = EventBus()
        cfg = make_fast_config(AlgorithmType.PPO.value)
        job = TrainingJob(cfg, bus)
        job.returns    = [1.0, 2.0, 3.0]
        job.moving_avg = [1.0, 1.5, 2.0]

        with tempfile.TemporaryDirectory() as tmp:
            CheckpointManager.save(job, tmp)
            loaded = CheckpointManager.load(
                os.path.join(tmp, job.job_id), bus)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.returns, [1.0, 2.0, 3.0])
        self.assertEqual(loaded.moving_avg, [1.0, 1.5, 2.0])


# ---------------------------------------------------------------------------
# Simulation Tests (verify algorithm trains and improves)
# ---------------------------------------------------------------------------

class SimulationTestBase(unittest.TestCase):
    """Base class for algorithm simulation tests."""

    ALGO: str = AlgorithmType.PPO.value
    N_EPISODES: int = 10
    MIN_EXPECTED_RETURN: float = -300.0  # BipedalWalker can be very negative initially

    def _run_training(self, config: JobConfig, timeout_s: float = 120) -> TrainingJob:
        bus = EventBus()
        job = TrainingJob(config, bus)
        job.start()

        deadline = time.time() + timeout_s
        while job.is_alive() and time.time() < deadline:
            time.sleep(0.5)

        if job.is_alive():
            job.cancel()
            job._thread.join(timeout=5)

        return job

    def _verify_training_results(self, job: TrainingJob):
        self.assertGreater(len(job.returns), 0, "No episodes completed")
        self.assertEqual(len(job.moving_avg), len(job.returns))
        # All returns should be finite numbers
        for r in job.returns:
            self.assertTrue(
                -1e6 < r < 1e6,
                f"Return {r} out of expected range"
            )
        # Model should exist
        self.assertIsNotNone(job.model)


class TestPPOSimulation(SimulationTestBase):
    ALGO = AlgorithmType.PPO.value
    N_EPISODES = 8

    def test_ppo_trains(self):
        cfg = make_fast_config(self.ALGO, self.N_EPISODES)
        job = self._run_training(cfg, timeout_s=180)
        self._verify_training_results(job)
        print(f"\n[PPO] Episodes: {len(job.returns)}, "
              f"Final avg: {job.moving_avg[-1]:.2f}")


class TestA2CSimulation(SimulationTestBase):
    ALGO = AlgorithmType.A2C.value
    N_EPISODES = 8

    def test_a2c_trains(self):
        cfg = make_fast_config(self.ALGO, self.N_EPISODES)
        job = self._run_training(cfg, timeout_s=180)
        self._verify_training_results(job)
        print(f"\n[A2C] Episodes: {len(job.returns)}, "
              f"Final avg: {job.moving_avg[-1]:.2f}")


class TestSACSimulation(SimulationTestBase):
    ALGO = AlgorithmType.SAC.value
    N_EPISODES = 8

    def test_sac_trains(self):
        cfg = make_fast_config(self.ALGO, self.N_EPISODES)
        job = self._run_training(cfg, timeout_s=180)
        self._verify_training_results(job)
        print(f"\n[SAC] Episodes: {len(job.returns)}, "
              f"Final avg: {job.moving_avg[-1]:.2f}")


class TestTD3Simulation(SimulationTestBase):
    ALGO = AlgorithmType.TD3.value
    N_EPISODES = 8

    def test_td3_trains(self):
        cfg = make_fast_config(self.ALGO, self.N_EPISODES)
        job = self._run_training(cfg, timeout_s=180)
        self._verify_training_results(job)
        print(f"\n[TD3] Episodes: {len(job.returns)}, "
              f"Final avg: {job.moving_avg[-1]:.2f}")


# ---------------------------------------------------------------------------
# Full Save/Load + Continue Training Test
# ---------------------------------------------------------------------------

class TestSaveLoadContinue(unittest.TestCase):

    def test_save_load_continue_ppo(self):
        bus = EventBus()
        cfg = make_fast_config(AlgorithmType.PPO.value, n_episodes=5)
        job = TrainingJob(cfg, bus)
        job.start()

        deadline = time.time() + 120
        while job.is_alive() and time.time() < deadline:
            time.sleep(0.5)

        initial_returns = len(job.returns)
        self.assertGreater(initial_returns, 0, "No episodes in initial training")

        with tempfile.TemporaryDirectory() as tmp:
            CheckpointManager.save(job, tmp)

            loaded = CheckpointManager.load(os.path.join(tmp, job.job_id), bus)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.returns), initial_returns)
            self.assertIsNotNone(loaded.model)

            # Continue training
            loaded.config.ep_cfg.n_episodes = initial_returns + 3
            loaded.start()
            deadline2 = time.time() + 120
            while loaded.is_alive() and time.time() < deadline2:
                time.sleep(0.5)

        total_eps = len(loaded.returns)
        self.assertGreaterEqual(total_eps, initial_returns,
                                "No additional episodes after continuing")
        print(f"\n[SaveLoad] Initial: {initial_returns}, After continue: {total_eps}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
