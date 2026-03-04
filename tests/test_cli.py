# First-party
import neural_lam
import neural_lam.create_graph
import neural_lam.train_model


def test_import():
    """This test just ensures that each cli entry-point can be imported for now,
    eventually we should test their execution too."""
    assert neural_lam is not None
    assert neural_lam.create_graph is not None
    assert neural_lam.train_model is not None


def test_train_model_parses_and_passes_use_all_ensemble_members(monkeypatch):
    captured = {}

    class DummyDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class DummyModel:
        def __init__(self, args, config, datastore):
            self.args = args
            self.config = config
            self.datastore = datastore

    class DummyTrainer:
        global_rank = 1

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            return None

    class DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        neural_lam.train_model,
        "load_config_and_datastore",
        lambda config_path: (object(), object()),
    )
    monkeypatch.setattr(neural_lam.train_model, "WeatherDataModule", DummyDataModule)
    monkeypatch.setitem(neural_lam.train_model.MODELS, "graph_lam", DummyModel)
    monkeypatch.setattr(neural_lam.train_model.utils, "setup_training_logger", lambda **kwargs: object())
    monkeypatch.setattr(
        neural_lam.train_model.utils,
        "init_training_logger_metrics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(neural_lam.train_model.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        neural_lam.train_model.pl.callbacks,
        "ModelCheckpoint",
        DummyCheckpoint,
    )
    monkeypatch.setattr(neural_lam.train_model.seed, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(neural_lam.train_model.torch.cuda, "is_available", lambda: False)

    neural_lam.train_model.main(
        [
            "--config_path",
            "dummy.yaml",
            "--model",
            "graph_lam",
            "--epochs",
            "1",
            "--use_all_ensemble_members",
        ]
    )

    assert captured["use_all_ensemble_members"] is True
