from pathlib import Path
import shutil
from typing import Any, Iterator
from unittest.mock import patch

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import xarray as xr
from cartopy import crs as ccrs
from matplotlib.testing.decorators import check_figures_equal

from neural_lam import vis
from tests.conftest import init_datastore_example

@pytest.fixture(autouse=True)
def mock_cartopy_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    from cartopy.mpl.geoaxes import GeoAxes
    monkeypatch.setattr(GeoAxes, "coastlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *args, **kwargs: None)

@pytest.fixture(autouse=True)
def close_all_figures_after_test() -> Iterator[None]:
    yield
    plt.close("all")

@pytest.fixture(scope="module", autouse=True)
def cleanup_matplotlib_result_images() -> Iterator[None]:
    yield
    result_images_dir = Path(__file__).resolve().parents[1] / "result_images" / "test_vis"
    shutil.rmtree(result_images_dir, ignore_errors=True)

def test_plot_prediction() -> None:
    datastore = init_datastore_example("dummydata")
    n_grid = datastore.num_grid_points

    da_pred = xr.DataArray(np.linspace(0.0, 1.0, n_grid))
    da_target = xr.DataArray(np.linspace(1.0, 2.0, n_grid))

    fig = vis.plot_prediction(
        datastore=datastore,
        da_prediction=da_pred,
        da_target=da_target,
        title="Test Prediction",
        boundary_alpha=None,
        crop_to_interior=False,
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3
    assert fig.axes[0].get_title() == "Ground Truth"
    assert fig.axes[1].get_title() == "Prediction"
    
    if hasattr(fig, "_suptitle") and fig._suptitle is not None:
        assert fig._suptitle.get_text() == "Test Prediction"

    assert len(fig.axes[0].collections) == 1
    assert len(fig.axes[1].collections) == 1

    vmin_p, vmax_p = fig.axes[0].collections[0].get_clim()
    vmin_t, vmax_t = fig.axes[1].collections[0].get_clim()
    assert vmin_p == vmin_t
    assert vmax_p == vmax_t

    vmin_expected = min(da_pred.min().item(), da_target.min().item())
    vmax_expected = max(da_pred.max().item(), da_target.max().item())
    assert vmin_p == pytest.approx(vmin_expected)
    assert vmax_p == pytest.approx(vmax_expected)

def test_plot_error_map() -> None:
    datastore = init_datastore_example("dummydata")
    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    d_f = len(var_names)
    pred_steps = 4

    errors = torch.arange(1, pred_steps * d_f + 1, dtype=torch.float32).reshape(pred_steps, d_f)

    fig = vis.plot_error_map(
        errors=errors,
        datastore=datastore,
        title="Test Error Map",
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]

    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (d_f, pred_steps)
    assert ax.get_xlabel() == "Lead time (h)"
    assert ax.get_title() == "Test Error Map"

    tick_labels_x = [t.get_text() for t in ax.get_xticklabels() if t.get_text() != ""]
    expected_x_labels = [str(datastore.step_length * (i + 1)) for i in range(pred_steps)]
    assert tick_labels_x == expected_x_labels

    tick_labels_y = [t.get_text() for t in ax.get_yticklabels() if t.get_text() != ""]
    expected_y_labels = [f"{name} ({unit})" for name, unit in zip(var_names, var_units)]
    assert tick_labels_y == expected_y_labels

    assert len(ax.texts) == pred_steps * d_f

def test_plot_spatial_error_crop_to_interior_changes_extent() -> None:
    datastore = init_datastore_example("dummydata")
    
    shape = getattr(datastore, "grid_shape", (10, 10))
    nx, ny = shape[0], shape[1]
    
    boundary_mask = np.ones((nx, ny), dtype=int)
    boundary_mask[2:-2, 2:-2] = 0
    n_grid = nx * ny
    datastore.ds["boundary_mask"] = xr.DataArray(boundary_mask.reshape(n_grid), dims=["grid_index"])
    
    lat_arr, lon_arr = datastore.get_lat_lon(category="state")
    lat_2d = lat_arr.numpy().reshape(nx, ny)
    lon_2d = lon_arr.numpy().reshape(nx, ny)
    
    interior_mask = (boundary_mask == 0)
    expected_lon_min = lon_2d[interior_mask].min()
    expected_lon_max = lon_2d[interior_mask].max()
    expected_lat_min = lat_2d[interior_mask].min()
    expected_lat_max = lat_2d[interior_mask].max()
    
    error = torch.ones(n_grid, dtype=torch.float32)

    with patch("cartopy.mpl.geoaxes.GeoAxes.set_extent", autospec=True) as set_extent_mock:
        fig = vis.plot_spatial_error(
            error=error,
            datastore=datastore,
            boundary_alpha=None,
            crop_to_interior=True,
        )
        
        set_extent_mock.assert_called_once()
        call_args = set_extent_mock.call_args
        extents = call_args[0][1]
        
        crs_arg = call_args[1].get("crs") if "crs" in call_args[1] else (call_args[0][2] if len(call_args[0]) > 2 else None)
        assert isinstance(crs_arg, ccrs.PlateCarree)
        assert extents[0] == pytest.approx(expected_lon_min)
        assert extents[1] == pytest.approx(expected_lon_max)
        assert extents[2] == pytest.approx(expected_lat_min)
        assert extents[3] == pytest.approx(expected_lat_max)

@check_figures_equal(tol=0, extensions=["png"])
def test_plot_error_map_check_figures_equal(fig_test: matplotlib.figure.Figure, fig_ref: matplotlib.figure.Figure) -> None:
    datastore = init_datastore_example("dummydata")
    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    d_f = len(var_names)
    pred_steps = 4

    errors = torch.arange(1, pred_steps * d_f + 1, dtype=torch.float32).reshape(pred_steps, d_f)

    with patch("neural_lam.vis.plt.subplots") as mock_subplots:
        ax_test = fig_test.subplots()
        mock_subplots.return_value = (fig_test, ax_test)
        
        vis.plot_error_map(
            errors=errors,
            datastore=datastore,
            title="Visual Compare",
        )
    errors_np = errors.T.cpu().numpy()
    row_maxes = np.max(errors_np, axis=1, keepdims=True)
    errors_norm = np.where(row_maxes > 0, errors_np / row_maxes, 0)
    
    with matplotlib.rc_context(vis.utils.fractional_plot_bundle(1)):
        fig_ref.set_size_inches(15, 10)
        ax_ref = fig_ref.subplots()
        
        ax_ref.imshow(errors_norm, cmap="OrRd", vmin=0, vmax=1.0, interpolation="none", aspect="auto", alpha=0.8)
        
        for i in range(d_f):
            for j in range(pred_steps):
                val = errors_np[i, j]
                text_str = f"{val:.3f}" if val < 9999 else f"{val:.2e}"
                ax_ref.text(j, i, text_str, ha="center", va="center", color="black")
                
        pred_hor_h = datastore.step_length * (np.arange(pred_steps) + 1)
        ax_ref.set_xticks(np.arange(pred_steps))
        ax_ref.set_xticklabels(pred_hor_h)
        ax_ref.set_xlabel("Lead time (h)")
        
        ax_ref.set_yticks(np.arange(d_f))
        ax_ref.set_yticklabels([f"{n} ({u})" for n, u in zip(var_names, var_units)], rotation=30)
        
        ax_ref.set_title("Visual Compare")
