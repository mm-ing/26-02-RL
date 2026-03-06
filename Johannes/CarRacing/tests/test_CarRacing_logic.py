import numpy as np

from CarRacing_logic import CarRacingTrainer, EnvConfig, TrainConfig, parse_hidden_layers


class DummySpace:
    def sample(self):
        return 0


class DummyEnv:
    def __init__(self):
        self.action_space = DummySpace()
        self.steps = 0

    def reset(self):
        self.steps = 0
        return np.zeros((4,)), {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= 5
        return np.zeros((4,)), 1.0, done, False, {}

    def render(self):
        return np.zeros((16, 16, 3), dtype=np.uint8)

    def close(self):
        return None


def test_parse_hidden_layers_single_value_to_symmetric_architecture():
    assert parse_hidden_layers("256") == [256, 256]


def test_parse_hidden_layers_csv_value_to_multi_layer_architecture():
    assert parse_hidden_layers("256,128,64") == [256, 128, 64]


def test_policy_mode_mapping():
    assert CarRacingTrainer.policy_is_continuous("SAC") is True
    assert CarRacingTrainer.policy_is_continuous("TD3") is True
    assert CarRacingTrainer.policy_is_continuous("DDQN") is False
    assert CarRacingTrainer.policy_is_continuous("QR-DQN") is False


def test_run_episode_returns_executed_steps(monkeypatch):
    trainer = CarRacingTrainer()
    monkeypatch.setattr(trainer.env_wrapper, "make_env", lambda **kwargs: DummyEnv())
    result = trainer.run_episode(model=None, max_steps=20, capture_frames=True, frame_stride=2)
    assert result["steps"] == 5
    assert result["reward"] == 5.0
    assert len(result["frames"]) >= 1


class DummyTrainEnv:
    def close(self):
        return None


class DummyModel:
    def __init__(self, trainer, cancel_on_first=False):
        self.trainer = trainer
        self.cancel_on_first = cancel_on_first
        self.learn_calls = 0
        self.policy = None

    def learn(self, total_timesteps, reset_num_timesteps, callback, progress_bar):
        self.learn_calls += 1
        target_episodes = max(1, int(getattr(callback, "total_episodes", 1)))
        for idx in range(target_episodes):
            callback.locals = {"infos": [{"episode": {"r": float(idx + 1), "l": int(total_timesteps)}}]}
            should_continue = callback._on_step()
            if self.cancel_on_first and idx == 0:
                self.trainer.cancel()
            if not should_continue:
                break
        return self

    def predict(self, obs, deterministic=True):
        return 0, None


def test_pause_resume_cancel_transitions_and_final_status(monkeypatch):
    events = []
    trainer = CarRacingTrainer(event_sink=lambda payload: events.append(payload))
    monkeypatch.setattr(trainer.env_wrapper, "make_env", lambda **kwargs: DummyTrainEnv())
    monkeypatch.setattr(trainer.factory, "create_model", lambda env, policy_name, params, device: DummyModel(trainer, cancel_on_first=True))

    trainer.set_paused(True)
    assert trainer.pause_event.is_set() is False
    trainer.set_paused(False)
    assert trainer.pause_event.is_set() is True

    config = TrainConfig(
        policy_name="SAC",
        episodes=5,
        max_steps=10,
        params={"gamma": 0.99, "learning_rate": 3e-4, "batch_size": 64},
        env_config=EnvConfig(),
        animation_on=False,
        run_id="run_cancel",
        session_id="session_cancel",
    )
    summary = trainer.train(config)

    assert summary["type"] == "training_done"
    assert summary["cancelled"] is True
    assert summary["episodes_done"] == 1
    assert events[-1]["type"] == "training_done"
    assert events[-1]["cancelled"] is True


def test_deterministic_eval_cadence_every_tenth_episode(monkeypatch):
    trainer = CarRacingTrainer()
    monkeypatch.setattr(trainer.env_wrapper, "make_env", lambda **kwargs: DummyTrainEnv())
    monkeypatch.setattr(trainer.factory, "create_model", lambda env, policy_name, params, device: DummyModel(trainer, cancel_on_first=False))

    eval_calls = []

    def fake_eval(model, max_steps):
        eval_calls.append(max_steps)
        return {"reward": 42.0, "steps": max_steps, "frames": [], "transitions": []}

    monkeypatch.setattr(trainer, "evaluate_policy", fake_eval)

    config = TrainConfig(
        policy_name="SAC",
        episodes=25,
        max_steps=10,
        params={"gamma": 0.99, "learning_rate": 3e-4, "batch_size": 64},
        env_config=EnvConfig(),
        animation_on=False,
        run_id="run_eval",
        session_id="session_eval",
    )
    summary = trainer.train(config)

    assert summary["cancelled"] is False
    assert summary["episodes_done"] == 25
    assert summary["eval_points"] == [(10, 42.0), (20, 42.0)]
    assert len(eval_calls) == 2
