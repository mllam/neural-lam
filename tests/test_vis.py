from pathlib import Path
import shutil
from typing import Any, Iterator
from unittest.mock import patch

# Third-party
from cartopy import crs as ccrs
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal
import numpy as np
import pytest
import torch
import xarray as xr

# First-party
from neural_lam import vis
from tests.conftest import init_datastore_example


@pytest.fixture(autouse=True)
def mock_cartopy_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Prevent cartopy from downloading Natural Earth map data during tests.
    Monkeypatches the GeoAxes methods used in vis.plot_on_axis.
    """
    from cartopy.mpl.geoaxes import GeoAxes

    monkeypatch.setattr(GeoAxes, "coastlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def close_all_figures_after_test() -> Iterator[None]:
    """Ensure test-created matplotlib figures are always cleaned up."""
    yield
    plt.close("all")


@pytest.fixture(scope="module", autouse=True)
def cleanup_matplotlib_result_images() -> Iterator[None]:
    """Remove matplotlib image-comparison artifacts for this test module."""
    result_images_dir = (
        Path(__file__).resolve().parents[1] / "result_images" / "test_vis"
    )
    result_images_dir.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(result_images_dir, ignore_errors=True)


def test_plot_prediction() -> None:
    """Check prediction plot structure, titles and shared color scaling."""
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

    ground_truth_ax, prediction_ax, _ = fig.axes
    assert ground_truth_ax.get_title() == "Ground Truth"
    assert prediction_ax.get_title() == "Prediction"
    assert fig._suptitle.get_text() == "Test Prediction"

    assert len(ground_truth_ax.collections) == 1
    assert len(prediction_ax.collections) == 1

    expected_vmin = float(min(da_pred.min(), da_target.min()))
    expected_vmax = float(max(da_pred.max(), da_target.max()))
    assert ground_truth_ax.collections[0].norm.vmin == expected_vmin
    assert ground_truth_ax.collections[0].norm.vmax == expected_vmax
    assert prediction_ax.collections[0].norm.vmin == expected_vmin
    assert prediction_ax.collections[0].norm.vmax == expected_vmax


def test_plot_error_map() -> None:
    """Check error heatmap content, labels and annotations."""
    datastore = init_datastore_example("dummydata")
    d_f = len(datastore.get_vars_names(category="state"))
    pred_steps = 4

    errors = torch.arange(1, pred_steps * d_f + 1, dtype=torch.float32).reshape(
        pred_steps, d_f
    )

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

    expected_x_ticklabels = [str(step) for step in range(1, pred_steps + 1)]
    actual_x_ticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert actual_x_ticklabels == expected_x_ticklabels

    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    expected_y_ticklabels = [
        f"{name} ({unit})" for name, unit in zip(var_names, var_units)
    ]
    actual_y_ticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert actual_y_ticklabels == expected_y_ticklabels

    assert len(ax.texts) == pred_steps * d_f


def test_plot_spatial_error_crop_to_interior_changes_extent() -> None:
    """Check interior cropping forwards interior lon/lat bounds to set_extent."""
    datastore = init_datastore_example("dummydata")
    n_grid = datastore.num_grid_points
    grid_shape = (datastore.grid_shape_state.x, datastore.grid_shape_state.y)

    boundary_mask = np.ones(grid_shape, dtype=int)
    boundary_mask[2:-2, 2:-2] = 0
    datastore.ds["boundary_mask"] = xr.DataArray(
        boundary_mask.reshape(n_grid), dims=["grid_index"]
    )
    # Clear any cached boundary_mask property so it reflects the updated dataset.
    datastore.__dict__.pop("boundary_mask", None)

    lats_lons = datastore.get_lat_lon("state")
    lons = lats_lons[:, 0].reshape(grid_shape)
    lats = lats_lons[:, 1].reshape(grid_shape)
    interior = boundary_mask == 0

    expected_min_lon = float(lons[interior].min())
    expected_max_lon = float(lons[interior].max())
    expected_min_lat = float(lats[interior].min())
    expected_max_lat = float(lats[interior].max())

    error = torch.linspace(0.0, 1.0, n_grid)
    with patch(
        "cartopy.mpl.geoaxes.GeoAxes.set_extent", autospec=True
    ) as set_extent_mock:
        fig = vis.plot_spatial_error(
            error=error,
            datastore=datastore,
            boundary_alpha=None,
            crop_to_interior=True,
        )

    assert set_extent_mock.call_count == 1
    called_extent = set_extent_mock.call_args.args[1]
    called_crs = set_extent_mock.call_args.kwargs["crs"]

    assert called_extent[0] == pytest.approx(expected_min_lon)
    assert called_extent[1] == pytest.approx(expected_max_lon)
    assert called_extent[2] == pytest.approx(expected_min_lat)
    assert called_extent[3] == pytest.approx(expected_max_lat)
    assert isinstance(called_crs, ccrs.PlateCarree)


@check_figures_equal(tol=0, extensions=["png"])
def test_plot_error_map_check_figures_equal(
    fig_test: matplotlib.figure.Figure, fig_ref: matplotlib.figure.Figure
) -> None:
    """Visual regression test for error heatmap using Matplotlib's built-in comparator."""
    datastore = init_datastore_example("dummydata")
    d_f = len(datastore.get_vars_names(category="state"))
    pred_steps = 4
    errors = torch.arange(1, pred_steps * d_f + 1, dtype=torch.float32).reshape(
        pred_steps, d_f
    )

    def subplots_on_test_figure(
        *args: Any, **kwargs: Any
    ) -> tuple[matplotlib.figure.Figure, Any]:
        figsize = kwargs.get("figsize")
        if figsize is not None:
            fig_test.set_size_inches(*figsize)
        ax = fig_test.subplots()
        return fig_test, ax

    with patch("neural_lam.vis.plt.subplots", side_effect=subplots_on_test_figure):
        vis.plot_error_map(
            errors=errors,
            datastore=datastore,
            title="Visual Compare",
        )

    errors_np = errors.T.cpu().numpy()
    d_f_ref, pred_steps_ref = errors_np.shape
    max_errors = errors_np.max(axis=1)
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    with matplotlib.rc_context(vis.utils.fractional_plot_bundle(1)):
        fig_ref.set_size_inches(15, 10)
        ax_ref = fig_ref.subplots()

        ax_ref.imshow(
            errors_norm,
            cmap="OrRd",
            vmin=0,
            vmax=1.0,
            interpolation="none",
            aspect="auto",
            alpha=0.8,
        )

        for (j, i), error in np.ndenumerate(errors_np):
            formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
            ax_ref.text(i, j, formatted_error, ha="center", va="center", usetex=False)

        label_size = 15
        ax_ref.set_xticks(np.arange(pred_steps_ref))
        pred_hor_i = np.arange(pred_steps_ref) + 1
        pred_hor_h = datastore.step_length * pred_hor_i
        ax_ref.set_xticklabels(pred_hor_h, size=label_size)
        ax_ref.set_xlabel("Lead time (h)", size=label_size)

        ax_ref.set_yticks(np.arange(d_f_ref))
        var_names = datastore.get_vars_names(category="state")
        var_units = datastore.get_vars_units(category="state")
        y_ticklabels = [
            f"{name} ({unit})" for name, unit in zip(var_names, var_units)
        ]
        ax_ref.set_yticklabels(y_ticklabels, rotation=30, size=label_size)
        ax_ref.set_title("Visual Compare", size=15)

