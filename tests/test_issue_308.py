# Standard library
import datetime

# Third-party
import pytest
import torch

# First-party
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import init_datastore_example

def test_create_dataarray_from_tensor_forcing():
    """
    Reproduce issue #308: create_dataarray_from_tensor fails for category="forcing"
    """
    datastore = init_datastore_example("dummydata")
    dataset = WeatherDataset(datastore=datastore, split="train")
    
    # Get some forcing data
    da_forcing = datastore.get_dataarray(category="forcing", split="train")
    # Shape of forcing is (time, grid_index, forcing_feature)
    forcing_tensor = torch.from_numpy(da_forcing.values).float()
    times = da_forcing.time.values
    
    # This should work but currently fails with AttributeError: 'DataArray' object has no attribute 'state_feature'
    da_forcing_new = dataset.create_dataarray_from_tensor(
        tensor=forcing_tensor,
        time=times,
        category="forcing"
    )
    
    assert da_forcing_new.dims == da_forcing.dims
    assert "forcing_feature" in da_forcing_new.dims
    assert "state_feature" not in da_forcing_new.dims

def test_create_dataarray_from_tensor_static():
    """
    Check if it also works for static data
    """
    datastore = init_datastore_example("dummydata")
    dataset = WeatherDataset(datastore=datastore, split="train")
    
    da_static = datastore.get_dataarray(category="static", split="train")
    static_tensor = torch.from_numpy(da_static.values).float()
    
    # Static data doesn't have time dim usually, but create_dataarray_from_tensor 
    # might expect one or handle its absence. 
    # In WeatherDataset.py, it expects time if tensor is 2D (grid_index, feat)
    
    da_static_new = dataset.create_dataarray_from_tensor(
        tensor=static_tensor,
        time=datetime.datetime(2020, 1, 1), # Provide dummy time
        category="static"
    )
    
    assert "static_feature" in da_static_new.dims
