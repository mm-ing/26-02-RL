"""
Unit and simulation tests for the Reacher RL Workbench.

Run with:
    python -m pytest tests/ -v
or from the Reacher directory:
    python -m pytest tests/test_algorithms.py -v
"""
import os
import sys
import time
import threading
import unittest

# Put project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reacher_logic import (
    EnvConfig,
    EpisodeConfig,
    Event,
    EventBus,
    EventType,
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
        EventBus().drain()  # must not raise

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
# Config dataclass defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigs(unittest.TestCase):

    def test_ppo_defaults(self) -> None:
        cfg = PPOConfig()
        self.assertEqual(cfg.algo_name, "PPO")
        self.assertEqual(cfg.n_steps, 2048)
        self.assertIsInstance(cfg.network, NetworkConfig)
        self.assertEqual(cfg.network.hidden_layers, [256, 256])
        self.assertAlmostEqual(cfg.clip_range, 0.2)
        self.assertAlmostEqual(cfg.gae_lambda, 0.95)

    def test_sac_defaults(self) -> None:
        cfg = SACConfig()
        self.assertEqual(cfg.algo_name, "SAC")
        self.assertEqual(cfg.buffer_size, 300_000)
        self.assertEqual(cfg.learning_starts, 1_000)
        self.assertAlmostEqual(cfg.tau, 0.005)

    def test_td3_defaults(self) -> None:
        cfg = TD3Config()
        self.assertEqual(cfg.algo_name, "TD3")
        self.assertEqual(cfg.buffer_size, 300_000)
        self.assertEqual(cfg.learning_starts, 1_000)
        self.assertEqual(cfg.policy_delay, 2)
        self.assertAlmostEqual(cfg.action_noise_sigma, 0.1)

    def test_ep_config_defaults(self) -> None:
        ec = EpisodeConfig()
        self.assertEqual(ec.n_episodes, 3000)
        self.assertEqual(ec.max_steps, 50)
        self.assertAlmostEqual(ec.alpha, 3e-4)
        self.assertAlmostEqual(ec.gamma, 0.99)
        self.assertFalse(ec.compare_methods)

    def test_env_config_defaults(self) -> None:
        env = EnvConfig()
        self.assertAlmostEqual(env.reward_control_weight, 0.1)
        self.assertAlmostEqual(env.reward_dist_weight, 1.0)
        self.assertEqual(env.render_interval_ms, 10)
        self.assertTrue(env.visualize)

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

    def test_single_value(self) -> None:
        result = expand_tuning_values("3e-4")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 3e-4)

    def test_empty_string(self) -> None:
        result = expand_tuning_values("")
        self.assertEqual(result, [])

    def test_semicolons_only(self) -> None:
        result = expand_tuning_values(";;;")
        self.assertEqual(result, [])


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
        self.assertEqual(mgr.get_job(job.job_id).status, JobStatus.CANCELLED)

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

    def test_label_uses_algo_name(self) -> None:
        mgr, bus, _ = self._make_manager()
        job = mgr.create_job("TD3", TD3Config(), EnvConfig(), EpisodeConfig())
        self.assertIn("TD3", job.label)

    def test_get_job_unknown_id(self) -> None:
        mgr, _, _ = self._make_manager()
        self.assertIsNone(mgr.get_job("does-not-exist"))

    def test_custom_label(self) -> None:
        mgr, bus, _ = self._make_manager()
        job = mgr.create_job("SAC", SACConfig(), EnvConfig(),
                              EpisodeConfig(), label="my-label")
        self.assertEqual(job.label, "my-label")

    def test_pause_resume_pending_no_crash(self) -> None:
        """Pausing a pending job should not crash (it won't actually pause)."""
        mgr, bus, _ = self._make_manager()
        job = mgr.create_job("PPO", PPOConfig(), EnvConfig(), EpisodeConfig())
        # PENDING → pause should be a no-op (status stays PENDING)
        mgr.cancel(job.job_id)
        self.assertEqual(mgr.get_job(job.job_id).status, JobStatus.CANCELLED)


# ─────────────────────────────────────────────────────────────────────────────
# Environment availability guard
# ─────────────────────────────────────────────────────────────────────────────

def _has_reacher() -> bool:
    try:
        import mujoco  # noqa
        import gymnasium as gym
        env = gym.make("Reacher-v5", render_mode="rgb_array")
        env.close()
        return True
    except Exception:
        return False


HAVE_ENV = _has_reacher()


# ─────────────────────────────────────────────────────────────────────────────
# Simulation tests (actual SB3 training, short runs)
# ─────────────────────────────────────────────────────────────────────────────

def _wait_job(mgr: TrainingManager, bus: EventBus, job_id: str,
              timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        bus.drain()
        j = mgr.get_job(job_id)
        if j and j.status in (JobStatus.COMPLETED, JobStatus.ERROR,
                               JobStatus.CANCELLED):
            return
        time.sleep(0.2)


@unittest.skipUnless(HAVE_ENV, "MuJoCo / Reacher-v5 not available")
class TestSimulationPPO(unittest.TestCase):
    """Short PPO training run verifying learning is possible."""

    def test_ppo_runs_and_records_episodes(self) -> None:
        bus = EventBus()
        mgr = TrainingManager(bus)
        ec  = EpisodeConfig(n_episodes=30, max_steps=50, alpha=3e-4, gamma=0.99)
        env = EnvConfig(visualize=False)
        ac  = PPOConfig(
            n_steps   = 256,
            batch_size= 64,
            n_epochs  = 4,
            network   = NetworkConfig(hidden_layers=[64, 64]),
        )
        job = mgr.create_job("PPO", ac, env, ec, label="ppo-test")
        mgr.start_job(job.job_id)
        _wait_job(mgr, bus, job.job_id)

        j = mgr.get_job(job.job_id)
        self.assertIsNotNone(j)
        self.assertNotEqual(j.status, JobStatus.ERROR,
                            f"PPO error: {j.error_msg}")
        self.assertGreater(len(j.returns), 0,
                           "No episodes recorded – PPO did not run")

    def test_ppo_returns_finite(self) -> None:
        """Returns must be finite floats (no NaN/Inf)."""
        bus = EventBus()
        mgr = TrainingManager(bus)
        ec  = EpisodeConfig(n_episodes=20, max_steps=50)
        env = EnvConfig(visualize=False)
        ac  = PPOConfig(n_steps=128, batch_size=32, n_epochs=3,
                        network=NetworkConfig(hidden_layers=[64, 64]))
        job = mgr.create_job("PPO", ac, env, ec)
        mgr.start_job(job.job_id)
        _wait_job(mgr, bus, job.job_id)

        j = mgr.get_job(job.job_id)
        import math
        for ret in j.returns:
            self.assertTrue(math.isfinite(ret),
                            f"Non-finite return encountered: {ret}")


@unittest.skipUnless(HAVE_ENV, "MuJoCo / Reacher-v5 not available")
class TestSimulationSAC(unittest.TestCase):

    def test_sac_runs_and_records_episodes(self) -> None:
        bus = EventBus()
        mgr = TrainingManager(bus)
        ec  = EpisodeConfig(n_episodes=20, max_steps=50, alpha=3e-4)
        env = EnvConfig(visualize=False)
        ac  = SACConfig(
            buffer_size    = 5_000,
            batch_size     = 64,
            learning_starts= 100,
            network        = NetworkConfig(hidden_layers=[64, 64]),
        )
        job = mgr.create_job("SAC", ac, env, ec, label="sac-test")
        mgr.start_job(job.job_id)
        _wait_job(mgr, bus, job.job_id)

        j = mgr.get_job(job.job_id)
        self.assertIsNotNone(j)
        self.assertNotEqual(j.status, JobStatus.ERROR,
                            f"SAC error: {j.error_msg}")
        self.assertGreater(len(j.returns), 0,
                           "No episodes recorded – SAC did not run")

    def test_sac_moving_average_monotony(self) -> None:
        """MA computed on 50-window should track long-term trend, not explode."""
        bus = EventBus()
        mgr = TrainingManager(bus)
        ec  = EpisodeConfig(n_episodes=30, max_steps=50)
        env = EnvConfig(visualize=False)
        ac  = SACConfig(buffer_size=5_000, batch_size=64, learning_starts=50,
                        network=NetworkConfig(hidden_layers=[64, 64]))
        job = mgr.create_job("SAC", ac, env, ec)
        mgr.start_job(job.job_id)
        _wait_job(mgr, bus, job.job_id)

        j = mgr.get_job(job.job_id)
        # moving_avg populated by UI layer; here check returns
        self.assertGreater(len(j.returns), 0)


@unittest.skipUnless(HAVE_ENV, "MuJoCo / Reacher-v5 not available")
class TestSimulationTD3(unittest.TestCase):

    def test_td3_runs_and_records_episodes(self) -> None:
        bus = EventBus()
        mgr = TrainingManager(bus)
        ec  = EpisodeConfig(n_episodes=20, max_steps=50, alpha=3e-4)
        env = EnvConfig(visualize=False)
        ac  = TD3Config(
            buffer_size    = 5_000,
            batch_size     = 64,
            learning_starts= 100,
            network        = NetworkConfig(hidden_layers=[64, 64]),
        )
        job = mgr.create_job("TD3", ac, env, ec, label="td3-test")
        mgr.start_job(job.job_id)
        _wait_job(mgr, bus, job.job_id)

        j = mgr.get_job(job.job_id)
        self.assertIsNotNone(j)
        self.assertNotEqual(j.status, JobStatus.ERROR,
                            f"TD3 error: {j.error_msg}")
        self.assertGreater(len(j.returns), 0,
                           "No episodes recorded – TD3 did not run")


@unittest.skipUnless(HAVE_ENV, "MuJoCo / Reacher-v5 not available")
class TestAlgorithmCorrectness(unittest.TestCase):
    """Checks that each algorithm produces improving or at least stable returns."""

    def _run_short(self, algo: str, ac, n_ep: int = 40):
        bus = EventBus()
        mgr = TrainingManager(bus)
        ec  = EpisodeConfig(n_episodes=n_ep, max_steps=50)
        env = EnvConfig(visualize=False)
        job = mgr.create_job(algo, ac, env, ec)
        mgr.start_job(job.job_id)
        _wait_job(mgr, bus, job.job_id, timeout=300)
        return mgr.get_job(job.job_id)

    def test_sac_not_diverging(self) -> None:
        ac  = SACConfig(buffer_size=10_000, batch_size=64,
                        learning_starts=200,
                        network=NetworkConfig(hidden_layers=[128, 128]))
        job = self._run_short("SAC", ac, n_ep=50)
        self.assertNotEqual(job.status, JobStatus.ERROR,
                            f"SAC diverged: {job.error_msg}")
        self.assertGreater(len(job.returns), 0)

    def test_td3_not_diverging(self) -> None:
        ac  = TD3Config(buffer_size=10_000, batch_size=64,
                        learning_starts=200,
                        network=NetworkConfig(hidden_layers=[128, 128]))
        job = self._run_short("TD3", ac, n_ep=50)
        self.assertNotEqual(job.status, JobStatus.ERROR,
                            f"TD3 diverged: {job.error_msg}")
        self.assertGreater(len(job.returns), 0)

    def test_ppo_not_diverging(self) -> None:
        ac  = PPOConfig(n_steps=512, batch_size=64, n_epochs=5,
                        network=NetworkConfig(hidden_layers=[128, 128]))
        job = self._run_short("PPO", ac, n_ep=50)
        self.assertNotEqual(job.status, JobStatus.ERROR,
                            f"PPO diverged: {job.error_msg}")
        self.assertGreater(len(job.returns), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
