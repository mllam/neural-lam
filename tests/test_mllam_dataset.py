from neural_lam.datastore import MLLAMDatastore
from neural_lam.weather_dataset import WeatherDataset


def test_mllam():
    config_path = "tests/datastore_configs/mllam.example.danra.yaml"
    datastore = MLLAMDatastore(config_path=config_path)
    dataset = WeatherDataset(datastore=datastore)