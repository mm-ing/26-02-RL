"""
Unit and simulation tests for the InvertedDoublePendulum RL Workbench.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_algorithms.py
"""
import sys
import os
import time
import queue
import threading
import unittest

# make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inverted_double_pendulum_logic import (
    EnvConfig,
    EpisodeConfig,
    EventBus,
    EventType,
    Event,
    JobStatus,
    NetworkConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
    TrainingManager,
    expand_tuning_values,
)


# ─────────────────────────────────────────────────────────────────────────────
# EventBus
# ─────────────────────────────────────────────────────────────────────────────

class TestEventBus(unittest.TestCase):

    def test_publish_and_drain(self) -> None:
        bus = EventBus()
        received: list = []
        bus.subscribe(lambda e: received.append(e))

        ev = Event(EventType.JOB_CREATED, "job-1", {"key": "val"})
        bus.publish(ev)
        bus.drain()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].type, EventType.JOB_CREATED)
        self.assertEqual(received[0].data["key"], "val")

    def test_drain_empty(self) -> None:
        bus = EventBus()
        # should not raise
        bus.drain()

    def test_multiple_subscribers(self) -> None:
        bus = EventBus()
        calls_a: list = []
        calls_b: list = []
        bus.subscribe(lambda e: calls_a.append(e.type))
        bus.subscribe(lambda e: calls_b.append(e.type))
        bus.publish(Event(EventType.JOB_STARTED, "j"))
        bus.drain()
        self.assertEqual(len(calls_a), 1)
        self.assertEqual(len(calls_b), 1)

    def test_thread_safety(self) -> None:
        bus = EventBus()
        received: list = []
        bus.subscribe(lambda e: received.append(1))

        def publish_many():
            for i in range(50):
                bus.publish(Event(EventType.STEP_COMPLETED, f"j{i}"))

        t = threading.Thread(target=publish_many)
        t.start()
        t.join()
        bus.drain()
        self.assertEqual(len(received), 50)


# ─────────────────────────────────────────────────────────────────────────────
# Config Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigs(unittest.TestCase):

    def test_ppo_defaults(self) -> None:
        cfg = PPOConfig()
        self.assertEqual(cfg.algo_name, "PPO")
        self.assertEqual(cfg.n_steps, 2048)
        self.assertIsInstance(cfg.network, NetworkConfig)
        self.assertEqual(cfg.network.hidden_layers, [256, 256])

    def test_sac_defaults(self) -> None:
        cfg = SACConfig()
        self.assertEqual(cfg.algo_name, "SAC")
        self.assertEqual(cfg.buffer_size, 300_000)
        self.assertEqual(cfg.learning_starts, 10_000)

    def test_td3_defaults(self) -> None:
        cfg = TD3Config()
        self.assertEqual(cfg.algo_name, "TD3")
        self.assertEqual(cfg.buffer_size, 300_000)
        self.assertEqual(cfg.learning_starts, 10_000)
        self.assertEqual(cfg.policy_delay, 2)

    def test_ep_config_defaults(self) -> None:
        ec = EpisodeConfig()
        self.assertEqual(ec.n_episodes, 3000)
        self.assertAlmostEqual(ec.alpha, 3e-4)
        self.assertAlmostEqual(ec.gamma, 0.99)

    def test_env_config_defaults(self) -> None:
        env = EnvConfig()
        self.assertAlmostEqual(env.healthy_reward, 10.0)
        self.assertAlmostEqual(env.reset_noise_scale, 0.1)

    def test_network_config(self) -> None:
        net = NetworkConfig(hidden_layers=[128, 64], activation="tanh")
        self.assertEqual(net.hidden_layers, [128, 64])
        self.assertEqual(net.activation, "tanh")


# ─────────────────────────────────────────────────────────────────────────────
# expand_tuning_values
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandTuning(unittest.TestCase):

    def test_floats(self) -> None:
        result = expand_tuning_values("1e-4;3e-4;1e-3")
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1e-4)
        self.assertAlmostEqual(result[2], 1e-3)

    def test_ints(self) -> None:
        result = expand_tuning_values("64;128;256")
        self.assertEqual(result, [64, 128, 256])

    def test_hidden_layers(self) -> None:
        result = expand_tuning_values("256,256;256,256,256;256,256,128")
        self.assertEqual(result[0], [256, 256])
        self.assertEqual(result[1], [256, 256, 256])
        self.assertEqual(result[2], [256, 256, 128])


# ─────────────────────────────────────────────────────────────────────────────
# TrainingManager (unit, no real training)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingManager(unittest.TestCase):

    def _make_manager(self):
        bus = EventBus()
        events: list = []
        bus.subscribe(lambda e: events.append(e.type))
        mgr = TrainingManager(bus)
        return mgr, bus, events

    def test_create_job(self) -> None:
        mgr, bus, events = self._make_manager()
        ec  = EpisodeConfig(n_episodes=10, max_steps=50)
        env = EnvConfig()
        ac  = PPOConfig()
        job = mgr.create_job("PPO", ac, env, ec, label="test-job")
        bus.drain()

        self.assertIsNotNone(job)
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertIn(EventType.JOB_CREATED, events)

    def test_remove_job(self) -> None:
        mgr, bus, events = self._make_manager()
        job = mgr.create_job("SAC", SACConfig(), EnvConfig(),
                              EpisodeConfig(n_episodes=5))
        mgr.remove(job.job_id)
        bus.drain()
        self.assertIn(EventType.JOB_REMOVED, events)
        self.assertIsNone(mgr.get_job(job.job_id))

    def test_cancel_pending_job(self) -> None:
        mgr, bus, events = self._make_manager()
        job = mgr.create_job("TD3", TD3Config(), EnvConfig(),
                              EpisodeConfig(n_episodes=5))
        mgr.cancel(job.job_id)
        bus.drain()
        self.assertEqual(mgr.get_job(job.job_id).status,
                         JobStatus.CANCELLED)

    def test_toggle_visibility(self) -> None:
        mgr, bus, _ = self._make_manager()
        job = mgr.create_job("PPO", PPOConfig(), EnvConfig(),
                              EpisodeConfig(n_episodes=5))
        self.assertTrue(job.visible)
        vis = mgr.toggle_visibility(job.job_id)
        self.assertFalse(vis)
        vis = mgr.toggle_visibility(job.job_id)
        self.assertTrue(vis)

    def test_jobs_property(self) -> None:
        mgr, _, _ = self._make_manager()
        self.assertEqual(len(mgr.jobs), 0)
        mgr.create_job("PPO", PPOConfig(), EnvConfig(), EpisodeConfig())
        mgr.create_job("SAC", SACConfig(), EnvConfig(), EpisodeConfig())
        self.assertEqual(len(mgr.jobs), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation tests (actual SB3 training, short run)
# ─────────────────────────────────────────────────────────────────────────────

def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa
        import gymnasium as gym
        env = gym.make("InvertedDoublePendulum-v5", render_mode="rgb_array")
        env.close()
        return True
    except Exception:
        return False


@unittest.skipUnless(_has_mujoco(), "MuJoCo / gymnasium not available")
class TestSimulationPPO(unittest.TestCase):
    """Short PPO training run on InvertedDoublePendulum-v5."""

    def test_ppo_learns(self) -> None:
        bus = EventBus()
        episodes_seen: list = []
        bus.subscribe(
            lambda e: episodes_seen.append(e.data.get("return", 0))
            if e.type == EventType.EPISODE_COMPLETED else None
        )
        mgr = TrainingManager(bus)
        ec  = EpisodeConfig(n_episodes=20, max_steps=200, alpha=3e-4, gamma=0.99)
        env = EnvConfig(healthy_reward=10.0, reset_noise_scale=0.1,
                        visualize=False)
        ac  = PPOConfig(n_steps=400, batch_size=64, n_epochs=4,
                        network=NetworkConfig(hidden_layers=[64, 64]))
        job = mgr.create_job("PPO", ac, env, ec, label="ppo-sim-test")
        mgr.start_job(job.job_id)

        deadline = time.time() + 120  # 2 min max
        while time.time() < deadline:
            bus.drain()
            j = mgr.get_job(job.job_id)
            if j and j.status in (JobStatus.COMPLETED, JobStatus.ERROR,
                                   JobStatus.CANCELLED):
                break
            time.sleep(0.1)

        bus.drain()
        j = mgr.get_job(job.job_id)
        self.assertIsNotNone(j)
        # Should have collected at least some episodes
        self.assertGreater(len(j.returns), 0,
                           "No episodes recorded – training did not run")
        self.assertNotEqual(j.status, JobStatus.ERROR,
                            f"Job finished with error: {j.error_msg}")


@unittest.skipUnless(_has_mujoco(), "MuJoCo / gymnasium not available")
class TestSimulationSAC(unittest.TestCase):

    def test_sac_runs_without_error(self) -> None:
        bus = EventBus()
        mgr = TrainingManager(bus)
        ec  = EpisodeConfig(n_episodes=15, max_steps=200, alpha=3e-4)
        env = EnvConfig(visualize=False)
        ac  = SACConfig(buffer_size=5000, batch_size=64, learning_starts=50,
                        network=NetworkConfig(hidden_layers=[64, 64]))
        job = mgr.create_job("SAC", ac, env, ec, label="sac-sim-test")
        mgr.start_job(job.job_id)

        deadline = time.time() + 120
        while time.time() < deadline:
            bus.drain()
            j = mgr.get_job(job.job_id)
            if j and j.status in (JobStatus.COMPLETED, JobStatus.ERROR,
                                   JobStatus.CANCELLED):
                break
            time.sleep(0.1)

        bus.drain()
        j = mgr.get_job(job.job_id)
        self.assertNotEqual(j.status, JobStatus.ERROR,
                            f"SAC error: {j.error_msg}")
        self.assertGreater(len(j.returns), 0)


@unittest.skipUnless(_has_mujoco(), "MuJoCo / gymnasium not available")
class TestSimulationTD3(unittest.TestCase):

    def test_td3_runs_without_error(self) -> None:
        bus = EventBus()
        mgr = TrainingManager(bus)
        ec  = EpisodeConfig(n_episodes=15, max_steps=200, alpha=3e-4)
        env = EnvConfig(visualize=False)
        ac  = TD3Config(buffer_size=5000, batch_size=64, learning_starts=50,
                        network=NetworkConfig(hidden_layers=[64, 64]))
        job = mgr.create_job("TD3", ac, env, ec, label="td3-sim-test")
        mgr.start_job(job.job_id)

        deadline = time.time() + 120
        while time.time() < deadline:
            bus.drain()
            j = mgr.get_job(job.job_id)
            if j and j.status in (JobStatus.COMPLETED, JobStatus.ERROR,
                                   JobStatus.CANCELLED):
                break
            time.sleep(0.1)

        bus.drain()
        j = mgr.get_job(job.job_id)
        self.assertNotEqual(j.status, JobStatus.ERROR,
                            f"TD3 error: {j.error_msg}")
        self.assertGreater(len(j.returns), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
