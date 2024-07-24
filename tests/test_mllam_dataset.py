# Third-party
import torch

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import MLLAMDatastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule, WeatherDataset


class ModelArgs:
    output_std = True
    loss = "mse"
    restore_opt = False
    n_example_pred = 1
    # XXX: this should be superfluous when we have already defined the model object
    graph = "multiscale"


def test_mllam():
    config_path = "tests/datastore_configs/mllam/example.danra.yaml"
    datastore = MLLAMDatastore(config_path=config_path)
    dataset = WeatherDataset(datastore=datastore)

    item = dataset[0]  # noqa

    data_module = WeatherDataModule(  # noqa
        ar_steps_train=3,
        ar_steps_eval=3,
        standardize=True,
        batch_size=2,
    )

    device_name = (  # noqa
        torch.device("cuda") if torch.cuda.is_available() else "cpu"
    )

    args = ModelArgs()

    create_graph_from_datastore(
        datastore=datastore,
        output_root_path="tests/datastore_configs/mllam/graph",
    )

    model = GraphLAM(  # noqa
        args=args,
        forcing_window_size=dataset.forcing_window_size,
        datastore=datastore,
    )
