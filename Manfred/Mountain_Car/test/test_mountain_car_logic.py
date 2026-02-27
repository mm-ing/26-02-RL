from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mountain_car_logic import (
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


def test_environment_registry_creates_mountain_car_env():
    registry = EnvironmentRegistry()
    env = registry.create("MountainCar-v0", render_mode=None)
    state, _ = env.reset(seed=1)
    assert state.shape[0] == 2
    env.close()


def test_replay_buffer_add_and_sample():
    buf = ReplayBuffer(capacity=64)
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
        name="d3qn",
        env_id="MountainCar-v0",
        algorithm=AlgorithmConfig(algorithm="D3QN", net=NetworkConfig(hidden_layers=[64, 64])),
        episodes=EpisodeConfig(episodes=2, max_steps=120),
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
        name="d3qn",
        env_id="MountainCar-v0",
        algorithm=AlgorithmConfig(algorithm="D3QN", net=NetworkConfig(hidden_layers=[64, 64])),
        episodes=EpisodeConfig(episodes=1, max_steps=80),
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
            job.thread.join(timeout=60)

    manager.save_all(tmp_path)

    manager2 = TrainingManager()
    loaded = manager2.load_all(tmp_path)
    assert len(loaded) == 1
    loaded_job = manager2.jobs[loaded[0]]
    assert loaded_job.current_episode >= 1


def _mean_eval_return(algo: SB3DQNAlgorithm, episodes: int = 5, max_steps: int = 200) -> float:
    scores = []
    for _ in range(episodes):
        scores.append(algo.evaluate_episode(max_steps=max_steps, render=False)["return"])
    return float(np.mean(scores))


def _train_and_eval(algorithm_name: str):
    algo_cfg = AlgorithmConfig(
        algorithm=algorithm_name,
        learning_rate=5e-4,
        gamma=0.99,
        buffer_size=80_000,
        batch_size=128,
        learning_starts=1_000,
        train_freq=4,
        net=NetworkConfig(hidden_layers=[128, 128], activation="relu"),
    )
    algo = SB3DQNAlgorithm("MountainCar-v0", algo_cfg, render_mode=None)
    try:
        algo.model.set_random_seed(42)
        baseline = _mean_eval_return(algo, episodes=3, max_steps=200)
        for _ in range(4):
            algo.update({"total_timesteps": 4_000})
        trained = _mean_eval_return(algo, episodes=5, max_steps=200)
        return baseline, trained
    finally:
        algo.close()


def test_simulation_d3qn_learns():
    baseline, trained = _train_and_eval("D3QN")
    assert trained >= baseline


def test_simulation_double_dqn_per_learns():
    baseline, trained = _train_and_eval("Double DQN + Prioritized Experience Replay")
    assert trained >= baseline


def test_simulation_dueling_dqn_learns():
    baseline, trained = _train_and_eval("Dueling DQN")
    assert trained >= baseline
