# Third-party
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import xarray as xr

# First-party
from neural_lam import vis
from conftest import init_datastore_example


@pytest.fixture(autouse=True)
def mock_cartopy_downloads(monkeypatch):
    """
    Prevent cartopy from downloading Natural Earth map data during tests.
    Monkeypatches the GeoAxes methods used in vis.plot_on_axis.
    """
    from cartopy.mpl.geoaxes import GeoAxes

    monkeypatch.setattr(GeoAxes, "coastlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *args, **kwargs: None)


def test_plot_prediction():
    """Verify plotting a spatial prediction map does not crash."""
    datastore = init_datastore_example("dummydata")
    n_grid = datastore.num_grid_points

    # Create random 1D grid arrays representing flattened predictions/targets
    da_pred = xr.DataArray(np.random.rand(n_grid))
    da_target = xr.DataArray(np.random.rand(n_grid))

    fig = vis.plot_prediction(
        datastore=datastore,
        da_prediction=da_pred,
        da_target=da_target,
        title="Test Prediction",
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_spatial_error():
    """Verify plotting a spatial error map does not crash."""
    datastore = init_datastore_example("dummydata")
    n_grid = datastore.num_grid_points

    # Error is passed as a torch Tensor in the main codebase
    error = torch.rand(n_grid)

    fig = vis.plot_spatial_error(
        error=error,
        datastore=datastore,
        title="Test Spatial Error",
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_error_map():
    """Verify the horizon error heatmap renders properly."""
    datastore = init_datastore_example("dummydata")
    d_f = len(datastore.get_vars_names(category="state"))
    pred_steps = 4

    # Shape expected: (pred_steps, d_f)
    errors = torch.rand(pred_steps, d_f)

    fig = vis.plot_error_map(
        errors=errors,
        datastore=datastore,
        title="Test Error Map",
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
