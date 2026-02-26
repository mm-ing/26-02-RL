from __future__ import annotations

from pathlib import Path

import numpy as np

from cartpole_logic import (
    AlgorithmConfig,
    EpisodeConfig,
    EnvironmentRegistry,
    JobConfig,
    NetworkConfig,
    ReplayBuffer,
    SB3DQNAlgorithm,
    TrainingManager,
    TuneConfig,
)


def test_environment_registry_creates_cartpole_env():
    registry = EnvironmentRegistry()
    env = registry.create("CartPole-v1", render_mode=None)
    state, _ = env.reset()
    assert state.shape[0] == 4
    env.close()


def test_replay_buffer_add_and_sample():
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        s = np.array([i, i + 1], dtype=np.float32)
        buf.add(s, i % 2, float(i), s + 1, i % 3 == 0)

    batch = buf.sample(8)
    assert batch["states"].shape == (8, 2)
    assert batch["actions"].shape == (8,)
    assert batch["dones"].shape == (8,)


def test_training_manager_add_pause_resume_remove():
    manager = TrainingManager()
    cfg = JobConfig(
        name="ddqn",
        env_id="CartPole-v1",
        algorithm=AlgorithmConfig(algorithm="DDQN", net=NetworkConfig(hidden_layers=[32, 32])),
        episodes=EpisodeConfig(episodes=2, max_steps=50),
    )
    job_id = manager.add_job(cfg)

    manager.start_job(job_id)
    manager.pause(job_id)
    assert manager.jobs[job_id].status == "paused"

    manager.resume(job_id)
    manager.cancel(job_id)
    manager.remove(job_id)
    assert job_id not in manager.jobs


def test_checkpoint_save_and_load(tmp_path: Path):
    manager = TrainingManager()
    cfg = JobConfig(
        name="ddqn",
        env_id="CartPole-v1",
        algorithm=AlgorithmConfig(algorithm="DDQN", net=NetworkConfig(hidden_layers=[32, 32])),
        episodes=EpisodeConfig(episodes=1, max_steps=20),
    )
    manager.create_jobs_for_compare_or_tuning(
        env_id=cfg.env_id,
        episode_cfg=cfg.episodes,
        base_algorithm=cfg.algorithm,
        compare_methods=False,
        tuning=TuneConfig(enabled=False),
    )
    manager.start_all_pending()

    for job in manager.jobs.values():
        if job.thread:
            job.thread.join(timeout=30)

    manager.save_all(tmp_path)

    manager2 = TrainingManager()
    loaded = manager2.load_all(tmp_path)
    assert len(loaded) == 1
    loaded_job = manager2.jobs[loaded[0]]
    assert loaded_job.current_episode >= 1


def _train_and_eval(algorithm_name: str):
    algo_cfg = AlgorithmConfig(
        algorithm=algorithm_name,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=20_000,
        batch_size=64,
        train_freq=4,
        net=NetworkConfig(hidden_layers=[64, 64], activation="relu"),
    )
    algo = SB3DQNAlgorithm("CartPole-v1", algo_cfg, render_mode=None)
    try:
        algo.model.set_random_seed(42)
        for _ in range(4):
            algo.update({"total_timesteps": 2_500})

        scores = []
        for _ in range(6):
            scores.append(algo.evaluate_episode(max_steps=500, render=False)["return"])
        return float(np.mean(scores))
    finally:
        algo.close()


def test_simulation_ddqn_learns():
    score = _train_and_eval("DDQN")
    assert score >= 25.0


def test_simulation_dueling_dqn_learns():
    score = _train_and_eval("Dueling DQN")
    assert score >= 25.0


def test_simulation_d3qn_learns():
    score = _train_and_eval("Double + Dueling DQN")
    assert score >= 25.0
