"""Tests for the high-level neural_lam Python API."""

# First-party
from neural_lam import api
from neural_lam import config as nlconfig
from tests.dummy_datastore import DummyDatastore


def test_api_train_and_evaluate(tmp_path):
    """Test full train and evaluate roundtrip using the Python API."""
    datastore = DummyDatastore(n_grid_points=100, n_timesteps=30)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )
    datastore._config = {}

    # We need to create a dummy graph first
    api.create_graph(
        datastore=datastore,
        config=config,
        name="test_graph",
        plot=False,
    )

    # Train
    train_run = api.train(
        datastore=datastore,
        config=config,
        model="graph_lam",
        graph="test_graph",
        epochs=1,
        ar_steps_train=1,
        val_interval=1,
        num_sanity_val_steps=0,
        batch_size=1,
        runs_root=str(tmp_path / "runs"),
        logger_run_name="test_run",
        val_steps_to_log=[1],
    )

    assert train_run.run_dir.exists()
    assert train_run.checkpoint_path is not None
    assert train_run.checkpoint_path.exists()

    # Evaluate
    eval_run = api.evaluate(
        datastore=datastore,
        config=config,
        model="graph_lam",
        graph="test_graph",
        load=str(train_run.checkpoint_path),
        ar_steps_eval=2,
        val_steps_to_log=[1, 2],
        runs_root=str(tmp_path / "runs"),
        logger_run_name="test_run",
    )

    assert eval_run.run_dir.exists()
