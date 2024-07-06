import torch

from neural_lam.datastore import MLLAMDatastore
from neural_lam.weather_dataset import WeatherDataset, WeatherDataModule
from neural_lam.models.graph_lam import GraphLAM

class ModelArgs:
    output_std = True
    loss = "mse"
    restore_opt = False
    n_example_pred = 1
    graph = "multiscale" # XXX: this should be superflous when we have already defined the model object


def test_mllam():
    config_path = "tests/datastore_configs/mllam/example.danra.yaml"
    datastore = MLLAMDatastore(config_path=config_path)
    dataset = WeatherDataset(datastore=datastore)
    
    item = dataset[0]

    data_module = WeatherDataModule(
        ar_steps_train=3,
        ar_steps_eval=3,
        standardize=True,
        batch_size=2,
    )
    
    import ipdb
    ipdb.set_trace()
    
    device_name = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    args = ModelArgs()
    
    model = GraphLAM(args=args, forcing_window_size=dataset.forcing_window_size, datastore=datastore)