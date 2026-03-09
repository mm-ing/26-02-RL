"""
Unit tests for the Ant RL Workbench.

Run with:
    python -m pytest Manfred/Ant/tests/ -v
or from the Ant directory:
    python -m pytest tests/ -v
"""
import sys
import os
import queue
import threading
import time
import unittest
from copy import deepcopy

# Ensure the project root (Ant folder) is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ant_logic import (
    CMAESConfig,
    EnvConfig,
    EpisodeConfig,
    Event,
    EventBus,
    EventType,
    JobStatus,
    NetworkConfig,
    SACConfig,
    TQCConfig,
    TrainingManager,
    expand_tuning_values,
    _ant_env_kwargs,
    _build_cmaes_net_class,
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
        bus.drain()   # must not raise

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
        t.join(timeout=2)
        bus.drain()
        self.assertEqual(len(received), 50)


# ─────────────────────────────────────────────────────────────────────────────
# Config Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigs(unittest.TestCase):

    def test_env_config_defaults(self) -> None:
        cfg = EnvConfig()
        self.assertAlmostEqual(cfg.forward_reward_weight, 1.0)
        self.assertAlmostEqual(cfg.ctrl_cost_weight, 0.5)
        self.assertAlmostEqual(cfg.contact_cost_weight, 5e-4)
        self.assertAlmostEqual(cfg.healthy_reward, 1.0)
        self.assertTrue(cfg.terminate_when_unhealthy)
        self.assertAlmostEqual(cfg.healthy_z_min, 0.2)
        self.assertAlmostEqual(cfg.healthy_z_max, 1.0)
        self.assertAlmostEqual(cfg.contact_force_min, -1.0)
        self.assertAlmostEqual(cfg.contact_force_max, 1.0)
        self.assertAlmostEqual(cfg.reset_noise_scale, 0.1)
        self.assertTrue(cfg.exclude_current_positions_from_observation)
        self.assertTrue(cfg.include_cfrc_ext_in_observation)
        self.assertEqual(cfg.render_interval_ms, 10)
        self.assertTrue(cfg.visualize)

    def test_episode_config_defaults(self) -> None:
        cfg = EpisodeConfig()
        self.assertEqual(cfg.n_episodes, 3000)
        self.assertEqual(cfg.max_steps, 1000)
        self.assertAlmostEqual(cfg.alpha, 3e-4)
        self.assertAlmostEqual(cfg.gamma, 0.99)
        self.assertFalse(cfg.compare_methods)

    def test_tqc_config_defaults(self) -> None:
        cfg = TQCConfig()
        self.assertEqual(cfg.algo_name, "TQC")
        self.assertEqual(cfg.buffer_size, 300_000)
        self.assertEqual(cfg.batch_size, 256)
        self.assertEqual(cfg.top_quantiles_to_drop, 2)
        self.assertEqual(cfg.n_quantiles, 25)
        self.assertEqual(cfg.n_critics, 2)
        self.assertIsInstance(cfg.network, NetworkConfig)

    def test_sac_config_defaults(self) -> None:
        cfg = SACConfig()
        self.assertEqual(cfg.algo_name, "SAC")
        self.assertIsInstance(cfg.network, NetworkConfig)

    def test_cmaes_config_defaults(self) -> None:
        cfg = CMAESConfig()
        self.assertEqual(cfg.algo_name, "CMA-ES")
        self.assertEqual(cfg.popsize, 50)
        self.assertAlmostEqual(cfg.stdev_init, 0.5)
        self.assertEqual(cfg.n_eval_episodes, 3)
        self.assertEqual(cfg.network.hidden_layers, [64, 64])
        self.assertEqual(cfg.network.activation, "tanh")

    def test_network_config_defaults(self) -> None:
        cfg = NetworkConfig()
        self.assertEqual(cfg.hidden_layers, [256, 256])
        self.assertEqual(cfg.activation, "relu")


# ─────────────────────────────────────────────────────────────────────────────
# Env kwargs helper
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvKwargs(unittest.TestCase):

    def test_no_render_mode_by_default(self) -> None:
        cfg = EnvConfig()
        kw = _ant_env_kwargs(cfg, render=False)
        self.assertNotIn("render_mode", kw)

    def test_render_mode_when_requested(self) -> None:
        cfg = EnvConfig()
        kw = _ant_env_kwargs(cfg, render=True)
        self.assertEqual(kw["render_mode"], "rgb_array")

    def test_healthy_z_range_is_tuple(self) -> None:
        cfg = EnvConfig(healthy_z_min=0.3, healthy_z_max=0.9)
        kw = _ant_env_kwargs(cfg)
        self.assertEqual(kw["healthy_z_range"], (0.3, 0.9))

    def test_contact_force_range_is_tuple(self) -> None:
        cfg = EnvConfig(contact_force_min=-2.0, contact_force_max=2.0)
        kw = _ant_env_kwargs(cfg)
        self.assertEqual(kw["contact_force_range"], (-2.0, 2.0))

    def test_all_keys_present(self) -> None:
        cfg = EnvConfig()
        kw = _ant_env_kwargs(cfg)
        expected_keys = {
            "forward_reward_weight",
            "ctrl_cost_weight",
            "contact_cost_weight",
            "healthy_reward",
            "terminate_when_unhealthy",
            "healthy_z_range",
            "contact_force_range",
            "reset_noise_scale",
            "exclude_current_positions_from_observation",
            "include_cfrc_ext_in_observation",
        }
        self.assertTrue(expected_keys.issubset(kw.keys()))


# ─────────────────────────────────────────────────────────────────────────────
# CMA-ES network builder
# ─────────────────────────────────────────────────────────────────────────────

class TestCMAESNetBuilder(unittest.TestCase):

    def test_output_shape(self) -> None:
        import torch
        net_cfg = NetworkConfig(hidden_layers=[32, 32], activation="tanh")
        PolicyNet = _build_cmaes_net_class(obs_dim=27, act_dim=8,
                                            net_cfg=net_cfg)
        net = PolicyNet()
        obs = torch.zeros(27)
        out = net(obs)
        self.assertEqual(out.shape, (8,))

    def test_output_bounded(self) -> None:
        """Final Tanh should bound all outputs to [-1, 1]."""
        import torch
        net_cfg = NetworkConfig(hidden_layers=[16], activation="relu")
        PolicyNet = _build_cmaes_net_class(obs_dim=10, act_dim=4,
                                            net_cfg=net_cfg)
        net = PolicyNet()
        obs = torch.ones(10) * 100.0   # extreme input
        out = net(obs)
        self.assertTrue((out.abs() <= 1.0 + 1e-6).all())

    def test_relu_activation(self) -> None:
        import torch
        net_cfg = NetworkConfig(hidden_layers=[16], activation="relu")
        PolicyNet = _build_cmaes_net_class(obs_dim=4, act_dim=2,
                                            net_cfg=net_cfg)
        net = PolicyNet()
        # just confirm it runs
        obs = torch.randn(4)
        out = net(obs)
        self.assertEqual(out.shape, (2,))

    def test_elu_activation(self) -> None:
        import torch
        net_cfg = NetworkConfig(hidden_layers=[8, 8], activation="elu")
        PolicyNet = _build_cmaes_net_class(obs_dim=5, act_dim=3,
                                            net_cfg=net_cfg)
        net = PolicyNet()
        obs = torch.randn(5)
        out = net(obs)
        self.assertEqual(out.shape, (3,))


# ─────────────────────────────────────────────────────────────────────────────
# TrainingManager
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingManager(unittest.TestCase):

    def _mgr(self):
        bus = EventBus()
        return TrainingManager(bus), bus

    def _sample_job_args(self):
        return (
            "SAC",
            SACConfig(),
            EnvConfig(),
            EpisodeConfig(n_episodes=5, max_steps=100),
        )

    def test_create_job(self) -> None:
        mgr, bus = self._mgr()
        events: list = []
        bus.subscribe(lambda e: events.append(e))

        job = mgr.create_job(*self._sample_job_args())
        bus.drain()

        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertTrue(any(e.type == EventType.JOB_CREATED for e in events))

    def test_create_multiple_jobs(self) -> None:
        mgr, bus = self._mgr()
        j1 = mgr.create_job(*self._sample_job_args())
        j2 = mgr.create_job(*self._sample_job_args())
        self.assertNotEqual(j1.job_id, j2.job_id)
        self.assertEqual(len(mgr.jobs), 2)

    def test_toggle_visibility(self) -> None:
        mgr, _ = self._mgr()
        job = mgr.create_job(*self._sample_job_args())
        self.assertTrue(job.visible)
        mgr.toggle_visibility(job.job_id)
        self.assertFalse(job.visible)
        mgr.toggle_visibility(job.job_id)
        self.assertTrue(job.visible)

    def test_cancel_pending_job(self) -> None:
        mgr, bus = self._mgr()
        events: list = []
        bus.subscribe(lambda e: events.append(e))
        job = mgr.create_job(*self._sample_job_args())
        mgr.cancel(job.job_id)
        bus.drain()
        self.assertEqual(job.status, JobStatus.CANCELLED)
        self.assertTrue(any(e.type == EventType.JOB_CANCELLED for e in events))

    def test_remove_job(self) -> None:
        mgr, bus = self._mgr()
        events: list = []
        bus.subscribe(lambda e: events.append(e))
        job = mgr.create_job(*self._sample_job_args())
        mgr.remove(job.job_id)
        bus.drain()
        self.assertEqual(len(mgr.jobs), 0)
        self.assertTrue(any(e.type == EventType.JOB_REMOVED for e in events))

    def test_get_job(self) -> None:
        mgr, _ = self._mgr()
        job = mgr.create_job(*self._sample_job_args())
        found = mgr.get_job(job.job_id)
        self.assertIs(found, job)

    def test_get_job_missing(self) -> None:
        mgr, _ = self._mgr()
        self.assertIsNone(mgr.get_job("nonexistent"))

    def test_jobs_property_is_copy(self) -> None:
        mgr, _ = self._mgr()
        mgr.create_job(*self._sample_job_args())
        jobs1 = mgr.jobs
        mgr.create_job(*self._sample_job_args())
        self.assertEqual(len(jobs1), 1)
        self.assertEqual(len(mgr.jobs), 2)

    def test_label_auto_assigned(self) -> None:
        mgr, _ = self._mgr()
        job = mgr.create_job("TQC", TQCConfig(), EnvConfig(),
                              EpisodeConfig(), label="")
        self.assertTrue(job.label.startswith("TQC-"))

    def test_label_custom(self) -> None:
        mgr, _ = self._mgr()
        job = mgr.create_job("SAC", SACConfig(), EnvConfig(),
                              EpisodeConfig(), label="my-run")
        self.assertEqual(job.label, "my-run")


# ─────────────────────────────────────────────────────────────────────────────
# expand_tuning_values
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandTuningValues(unittest.TestCase):

    def test_float_scalars(self) -> None:
        result = expand_tuning_values("1e-4;3e-4;1e-3")
        self.assertAlmostEqual(result[0], 1e-4)
        self.assertAlmostEqual(result[1], 3e-4)
        self.assertAlmostEqual(result[2], 1e-3)

    def test_int_scalar(self) -> None:
        result = expand_tuning_values("64;128;256")
        self.assertEqual(result, [64, 128, 256])

    def test_hidden_layers(self) -> None:
        result = expand_tuning_values("64,64;128,128")
        self.assertEqual(result[0], [64, 64])
        self.assertEqual(result[1], [128, 128])

    def test_mixed(self) -> None:
        result = expand_tuning_values("3e-4;64,64")
        self.assertAlmostEqual(result[0], 3e-4)
        self.assertEqual(result[1], [64, 64])

    def test_empty_string(self) -> None:
        result = expand_tuning_values("")
        self.assertEqual(result, [])

    def test_single_value(self) -> None:
        result = expand_tuning_values("0.001")
        self.assertAlmostEqual(result[0], 0.001)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: SAC short training smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestSACTraining(unittest.TestCase):
    """
    Smoke test: verify SAC completes at least 1 episode on Ant-v5.
    Requires gymnasium[mujoco] and stable-baselines3.
    Skipped if dependencies are missing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import gymnasium as gym
            import stable_baselines3
            env = gym.make("Ant-v5")
            env.close()
        except Exception as e:
            raise unittest.SkipTest(f"Ant-v5 not available: {e}")

    def test_sac_runs_episodes(self) -> None:
        bus = EventBus()
        mgr = TrainingManager(bus)

        ep_events: list = []
        done_events: list = []
        error_events: list = []

        def on_event(ev):
            if ev.type == EventType.EPISODE_COMPLETED:
                ep_events.append(ev)
            elif ev.type == EventType.JOB_COMPLETED:
                done_events.append(ev)
            elif ev.type == EventType.JOB_ERROR:
                error_events.append(ev)

        bus.subscribe(on_event)

        n_episodes = 3
        job = mgr.create_job(
            "SAC",
            SACConfig(
                buffer_size=2_000,
                batch_size=32,
                learning_starts=100,
                train_freq=1,
                gradient_steps=1,
                network=NetworkConfig(hidden_layers=[32, 32]),
            ),
            EnvConfig(visualize=False),
            EpisodeConfig(n_episodes=n_episodes, max_steps=200),
        )
        mgr.start_job(job.job_id)

        # Wait for training to complete (timeout 120 s for 3 short episodes)
        deadline = time.time() + 120
        while time.time() < deadline:
            bus.drain()
            if done_events or error_events:
                break
            time.sleep(0.2)

        bus.drain()

        self.assertEqual(error_events, [], msg=f"SAC error: {[e.data for e in error_events]}")
        self.assertGreater(len(ep_events), 0, "SAC produced no episode events")


# ─────────────────────────────────────────────────────────────────────────────
# Integration: TQC short training smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestTQCTraining(unittest.TestCase):
    """Smoke test for TQC. Skipped if sb3-contrib not available."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import gymnasium as gym
            import sb3_contrib
            env = gym.make("Ant-v5")
            env.close()
        except Exception as e:
            raise unittest.SkipTest(f"TQC or Ant-v5 not available: {e}")

    def test_tqc_runs_episodes(self) -> None:
        bus = EventBus()
        mgr = TrainingManager(bus)

        ep_events:    list = []
        done_events:  list = []
        error_events: list = []

        def on_event(ev):
            if ev.type == EventType.EPISODE_COMPLETED:
                ep_events.append(ev)
            elif ev.type == EventType.JOB_COMPLETED:
                done_events.append(ev)
            elif ev.type == EventType.JOB_ERROR:
                error_events.append(ev)

        bus.subscribe(on_event)

        job = mgr.create_job(
            "TQC",
            TQCConfig(
                buffer_size=2_000,
                batch_size=32,
                learning_starts=100,
                train_freq=1,
                gradient_steps=1,
                n_quantiles=10,
                n_critics=2,
                top_quantiles_to_drop=1,
                network=NetworkConfig(hidden_layers=[32, 32]),
            ),
            EnvConfig(visualize=False),
            EpisodeConfig(n_episodes=3, max_steps=200),
        )
        mgr.start_job(job.job_id)

        deadline = time.time() + 120
        while time.time() < deadline:
            bus.drain()
            if done_events or error_events:
                break
            time.sleep(0.2)

        bus.drain()

        self.assertEqual(error_events, [], msg=f"TQC error: {[e.data for e in error_events]}")
        self.assertGreater(len(ep_events), 0, "TQC produced no episode events")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
