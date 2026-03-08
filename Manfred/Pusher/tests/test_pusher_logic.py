"""
Unit tests for pusher_logic.py
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import threading
import time
import numpy as np

from pusher_logic import (
    ALGO_CONFIGS,
    CheckpointManager,
    DDPGConfig,
    EnvConfig,
    EpisodeConfig,
    Event,
    EventBus,
    EventType,
    JobStatus,
    NetworkConfig,
    SACConfig,
    TD3Config,
    TrainingJob,
    TrainingManager,
    expand_tuning_values,
    _activation_fn,
    _make_policy_kwargs,
)


# ─────────────────────────────────────────────────────────────────────────────
# EventBus tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEventBus:

    def test_subscribe_and_drain(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda e: received.append(e))
        bus.publish(Event(EventType.JOB_CREATED, "job1", {}))
        bus.publish(Event(EventType.JOB_STARTED, "job1", {}))
        bus.drain()
        assert len(received) == 2
        assert received[0].type == EventType.JOB_CREATED
        assert received[1].type == EventType.JOB_STARTED

    def test_drain_empty(self):
        bus = EventBus()
        # Should not raise
        bus.drain()

    def test_thread_safe_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda e: received.append(e.job_id))

        def publish_many():
            for i in range(50):
                bus.publish(Event(EventType.EPISODE_COMPLETED, f"job{i}", {}))

        t = threading.Thread(target=publish_many)
        t.start()
        t.join()
        bus.drain()
        assert len(received) == 50


# ─────────────────────────────────────────────────────────────────────────────
# Config Dataclass tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigs:

    def test_env_config_defaults(self):
        cfg = EnvConfig()
        assert cfg.reward_control_weight == 0.1
        assert cfg.reward_near_weight    == 0.5
        assert cfg.reward_dist_weight    == 1.0
        assert cfg.render_interval_ms   == 10
        assert cfg.visualize            is True

    def test_episode_config_defaults(self):
        cfg = EpisodeConfig()
        assert cfg.n_episodes == 3000
        assert cfg.max_steps  == 500
        assert cfg.alpha      == pytest.approx(3e-4)
        assert cfg.gamma      == pytest.approx(0.99)

    def test_sac_config_defaults(self):
        cfg = SACConfig()
        assert cfg.algo_name   == "SAC"
        assert cfg.buffer_size == 300_000
        assert cfg.batch_size  == 256
        assert cfg.ent_coef    == "auto"

    def test_td3_config_defaults(self):
        cfg = TD3Config()
        assert cfg.algo_name    == "TD3"
        assert cfg.policy_delay == 2

    def test_ddpg_config_defaults(self):
        cfg = DDPGConfig()
        assert cfg.algo_name  == "DDPG"
        assert cfg.tau        == pytest.approx(0.005)

    def test_network_config_defaults(self):
        cfg = NetworkConfig()
        assert cfg.hidden_layers == [256, 256]
        assert cfg.activation    == "relu"

    def test_algo_configs_registry(self):
        assert "SAC"  in ALGO_CONFIGS
        assert "TD3"  in ALGO_CONFIGS
        assert "DDPG" in ALGO_CONFIGS


# ─────────────────────────────────────────────────────────────────────────────
# Network helper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNetworkHelpers:

    def test_activation_fn_relu(self):
        import torch.nn as nn
        fn = _activation_fn("relu")
        assert fn == nn.ReLU

    def test_activation_fn_tanh(self):
        import torch.nn as nn
        fn = _activation_fn("tanh")
        assert fn == nn.Tanh

    def test_activation_fn_elu(self):
        import torch.nn as nn
        fn = _activation_fn("elu")
        assert fn == nn.ELU

    def test_activation_fn_unknown_defaults_relu(self):
        import torch.nn as nn
        fn = _activation_fn("unknown")
        assert fn == nn.ReLU

    def test_make_policy_kwargs(self):
        net = NetworkConfig(hidden_layers=[128, 64], activation="tanh")
        pk  = _make_policy_kwargs(net)
        import torch.nn as nn
        assert pk["net_arch"]      == [128, 64]
        assert pk["activation_fn"] == nn.Tanh


# ─────────────────────────────────────────────────────────────────────────────
# TrainingManager tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingManager:

    def _make_manager(self):
        bus = EventBus()
        mgr = TrainingManager(bus)
        return mgr, bus

    def test_create_job(self):
        mgr, bus = self._make_manager()
        job = mgr.create_job(
            "SAC", SACConfig(),
            EnvConfig(), EpisodeConfig(),
        )
        assert job.job_id in [j.job_id for j in mgr.jobs]
        assert job.status == JobStatus.PENDING

    def test_create_job_custom_label(self):
        mgr, _ = self._make_manager()
        job = mgr.create_job(
            "TD3", TD3Config(),
            EnvConfig(), EpisodeConfig(),
            label="my-td3-run",
        )
        assert job.label == "my-td3-run"

    def test_toggle_visibility(self):
        mgr, _ = self._make_manager()
        job = mgr.create_job("DDPG", DDPGConfig(), EnvConfig(), EpisodeConfig())
        assert job.visible is True
        mgr.toggle_visibility(job.job_id)
        assert job.visible is False
        mgr.toggle_visibility(job.job_id)
        assert job.visible is True

    def test_cancel_job(self):
        mgr, bus = self._make_manager()
        job = mgr.create_job("SAC", SACConfig(), EnvConfig(), EpisodeConfig())
        # Manually set running so cancel works
        job.status = JobStatus.RUNNING
        mgr.cancel(job.job_id)
        assert job.status == JobStatus.CANCELLED

    def test_remove_job(self):
        mgr, bus = self._make_manager()
        events = []
        bus.subscribe(lambda e: events.append(e))
        job = mgr.create_job("SAC", SACConfig(), EnvConfig(), EpisodeConfig())
        jid = job.job_id
        mgr.remove(jid)
        assert mgr.get_job(jid) is None
        bus.drain()
        removed = [e for e in events if e.type == EventType.JOB_REMOVED]
        assert len(removed) == 1

    def test_get_job_not_found(self):
        mgr, _ = self._make_manager()
        assert mgr.get_job("nonexistent-id") is None

    def test_multiple_jobs(self):
        mgr, _ = self._make_manager()
        for algo in ["SAC", "TD3", "DDPG"]:
            cfg = ALGO_CONFIGS[algo]()
            mgr.create_job(algo, cfg, EnvConfig(), EpisodeConfig())
        assert len(mgr.jobs) == 3


# ─────────────────────────────────────────────────────────────────────────────
# expand_tuning_values tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandTuningValues:

    def test_float_values(self):
        result = expand_tuning_values("1e-4;3e-4;1e-3")
        assert result == pytest.approx([1e-4, 3e-4, 1e-3])

    def test_int_value(self):
        result = expand_tuning_values("256")
        assert result == [256]
        assert isinstance(result[0], int)

    def test_hidden_layer_list(self):
        result = expand_tuning_values("256,256;512,256,128")
        assert result == [[256, 256], [512, 256, 128]]

    def test_mixed_values(self):
        result = expand_tuning_values("1e-4;256,256;0.99")
        assert result[0] == pytest.approx(1e-4)
        assert result[1] == [256, 256]
        assert result[2] == pytest.approx(0.99)

    def test_empty_string(self):
        result = expand_tuning_values("")
        assert result == []

    def test_single_value(self):
        result = expand_tuning_values("0.005")
        assert result == pytest.approx([0.005])


# ─────────────────────────────────────────────────────────────────────────────
# Simulation test: SAC learns on Pusher-v5 (short run)
# ─────────────────────────────────────────────────────────────────────────────

class TestSACSimulation:
    """Verify SAC can complete a short training run without errors."""

    def test_sac_training_completes(self):
        pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")

        bus    = EventBus()
        mgr    = TrainingManager(bus)
        events = []
        bus.subscribe(lambda e: events.append(e))

        ep_cfg  = EpisodeConfig(n_episodes=5, max_steps=50)
        env_cfg = EnvConfig(visualize=False)
        sac_cfg = SACConfig(
            buffer_size=2000,
            batch_size=64,
            learning_starts=200,
        )

        job = mgr.create_job("SAC", sac_cfg, env_cfg, ep_cfg)
        mgr.start_job(job.job_id)

        # Wait up to 120 s for completion
        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(0.5)
            bus.drain()
            j = mgr.get_job(job.job_id)
            if j and j.status in (JobStatus.COMPLETED, JobStatus.ERROR,
                                  JobStatus.CANCELLED):
                break

        j = mgr.get_job(job.job_id)
        assert j is not None
        assert j.status == JobStatus.COMPLETED, \
            f"Expected COMPLETED, got {j.status}; error: {j.error_msg}"
        assert len(j.returns) > 0


class TestTD3Simulation:
    """Verify TD3 can complete a short training run without errors."""

    def test_td3_training_completes(self):
        pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")

        bus    = EventBus()
        mgr    = TrainingManager(bus)
        events = []
        bus.subscribe(lambda e: events.append(e))

        ep_cfg  = EpisodeConfig(n_episodes=5, max_steps=50)
        env_cfg = EnvConfig(visualize=False)
        td3_cfg = TD3Config(
            buffer_size=2000,
            batch_size=64,
            learning_starts=200,
        )

        job = mgr.create_job("TD3", td3_cfg, env_cfg, ep_cfg)
        mgr.start_job(job.job_id)

        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(0.5)
            bus.drain()
            j = mgr.get_job(job.job_id)
            if j and j.status in (JobStatus.COMPLETED, JobStatus.ERROR,
                                  JobStatus.CANCELLED):
                break

        j = mgr.get_job(job.job_id)
        assert j is not None
        assert j.status == JobStatus.COMPLETED, \
            f"Expected COMPLETED, got {j.status}; error: {j.error_msg}"
        assert len(j.returns) > 0


class TestDDPGSimulation:
    """Verify DDPG can complete a short training run without errors."""

    def test_ddpg_training_completes(self):
        pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")

        bus    = EventBus()
        mgr    = TrainingManager(bus)
        events = []
        bus.subscribe(lambda e: events.append(e))

        ep_cfg   = EpisodeConfig(n_episodes=5, max_steps=50)
        env_cfg  = EnvConfig(visualize=False)
        ddpg_cfg = DDPGConfig(
            buffer_size=2000,
            batch_size=64,
            learning_starts=200,
            gradient_steps=1,
        )

        job = mgr.create_job("DDPG", ddpg_cfg, env_cfg, ep_cfg)
        mgr.start_job(job.job_id)

        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(0.5)
            bus.drain()
            j = mgr.get_job(job.job_id)
            if j and j.status in (JobStatus.COMPLETED, JobStatus.ERROR,
                                  JobStatus.CANCELLED):
                break

        j = mgr.get_job(job.job_id)
        assert j is not None
        assert j.status == JobStatus.COMPLETED, \
            f"Expected COMPLETED, got {j.status}; error: {j.error_msg}"
        assert len(j.returns) > 0
