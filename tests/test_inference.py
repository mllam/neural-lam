# Third-party
import pytest

# First-party
from neural_lam.train_model import main as train_model_main
from tests.conftest import init_datastore_example


@pytest.mark.dependency(depends=["test_training"])
def test_inference(request):
    """
    Run inference on a trained model and save the results to a zarr dataset
    through the command line interface.

    NB: This test will need refactoring once we clean up the command line
    interface
    """
    datastore = init_datastore_example("mdp")

    # NB: this is brittle and should be refactored when the command line
    # interface is cleaned up so that tests point to neural-lam config files
    # rather than datastore config files
    nl_config_path = datastore.root_path / "config.yaml"

    # fetch the path to the trained model that was saved by the training test
    model_path = request.config.cache.get("model_checkpoint_path", None)
    if model_path is None:
        raise Exception("training test must be run first")

    args = [
        "--config_path",
        nl_config_path,
        "--model",
        "graph_lam",
        "--eval",
        "test",
        "--load",
        model_path,
        "--hidden_dim",
        "4",
        "--hidden_layers",
        "1",
        "--processor_layers",
        "2",
        "--mesh_aggr",
        "sum",
        "--lr",
        "1.0e-3",
        "--val_steps_to_log",
        "1",
        "3",
        "--num_past_forcing_steps",
        "1",
        "--num_future_forcing_steps",
        "1",
        "--n_example_pred",
        "1",
        "--graph",
        "1level",
        "--save_eval_to_zarr_path",
        "state_test.zarr",
    ]

    train_model_main(args)
