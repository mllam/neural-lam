"""List of methods and attributes that should be implemented in a subclass of
`` (these are all decorated with `@abc.abstractmethod`):

- `root_path` (property): Root path of the datastore.
- `step_length` (property): Length of the time step in hours.
- `grid_shape_state` (property): Shape of the grid for the state variables.
- `get_xy` (method): Return the x, y coordinates of the dataset.
- `coords_projection` (property): Projection object for the coordinates.
- `get_vars_units` (method): Get the units of the variables in the given
      category.
- `get_vars_names` (method): Get the names of the variables in the given
      category.
- `get_vars_long_names` (method): Get the long names of the variables in
      the given category.
- `get_num_data_vars` (method): Get the number of data variables in the
      given category.
- `get_normalization_dataarray` (method): Return the normalization
      dataarray for the given category.
- `get_dataarray` (method): Return the processed data (as a single
      `xr.DataArray`) for the given category and test/train/val-split.
- `boundary_mask` (property): Return the boundary mask for the dataset,
      with spatial dimensions stacked.
- `config` (property): Return the configuration of the datastore.

In addition BaseRegularGridDatastore must have the following methods and
attributes:
- `get_xy_extent` (method): Return the extent of the x, y coordinates for a
        given category of data.
- `get_xy` (method): Return the x, y coordinates of the dataset.
- `coords_projection` (property): Projection object for the coordinates.
- `grid_shape_state` (property): Shape of the grid for the state variables.
- `stack_grid_coords` (method): Stack the grid coordinates of the dataset

"""

# Standard library
import collections
import dataclasses
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import numpy as np
import pytest
import torch
import xarray as xr

# First-party
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.datastore.plot_example import plot_example_from_datastore
from tests.conftest import init_datastore_example


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_root_path(datastore_name):
    """Check that the `datastore.root_path` property is implemented."""
    datastore = init_datastore_example(datastore_name)
    assert isinstance(datastore.root_path, Path)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_config(datastore_name):
    """Check that the `datastore.config` property is implemented."""
    datastore = init_datastore_example(datastore_name)
    # check the config is a mapping or a dataclass
    config = datastore.config
    assert isinstance(
        config, collections.abc.Mapping
    ) or dataclasses.is_dataclass(config)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_step_length(datastore_name):
    """Check that the `datastore.step_length` property is implemented."""
    datastore = init_datastore_example(datastore_name)
    step_length = datastore.step_length
    assert isinstance(step_length, int)
    assert step_length > 0


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_datastore_grid_xy(datastore_name):
    """Use the `datastore.get_xy` method to get the x, y coordinates of the
    dataset and check that the shape is correct against the `da
    tastore.grid_shape_state` property."""
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            "Skip grid_shape_state test for non-regular grid datastores"
        )

    # check the shapes of the xy grid
    grid_shape = datastore.grid_shape_state
    nx, ny = grid_shape.x, grid_shape.y
    for stacked in [True, False]:
        xy = datastore.get_xy("static", stacked=stacked)
        if stacked:
            assert xy.shape == (nx * ny, 2)
        else:
            assert xy.shape == (nx, ny, 2)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_vars(datastore_name):
    """Check that results of.

    - `datastore.get_vars_units`
    - `datastore.get_vars_names`
    - `datastore.get_vars_long_names`
    - `datastore.get_num_data_vars`

    are consistent (as in the number of variables are the same) and that the
    return types of each are correct.

    """
    datastore = init_datastore_example(datastore_name)

    for category in ["state", "forcing", "static"]:
        units = datastore.get_vars_units(category)
        names = datastore.get_vars_names(category)
        long_names = datastore.get_vars_long_names(category)
        num_vars = datastore.get_num_data_vars(category)

        assert len(units) == len(names) == num_vars
        assert isinstance(units, list)
        assert isinstance(names, list)
        assert isinstance(long_names, list)
        assert isinstance(num_vars, int)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_normalization_dataarray(datastore_name):
    """Check that the `datastore.get_normalization_dataarray` method is
    implemented."""
    datastore = init_datastore_example(datastore_name)

    for category in ["state", "forcing", "static"]:
        ds_stats = datastore.get_standardization_dataarray(category=category)

        # check that the returned object is an xarray DataArray
        # and that it has the correct variables
        assert isinstance(ds_stats, xr.Dataset)

        if category == "state":
            ops = [
                "mean",
                "std",
                "diff_mean_standardized",
                "diff_std_standardized",
            ]
        elif category == "forcing":
            ops = ["mean", "std"]
        elif category == "static":
            ops = []
        else:
            raise NotImplementedError(category)

        for op in ops:
            var_name = f"{category}_{op}"
            assert var_name in ds_stats.data_vars
            da_val = ds_stats[var_name]
            assert set(da_val.dims) == {f"{category}_feature"}


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_dataarray(datastore_name):
    """Check that the `datastore.get_dataarray` method is implemented.

    And that it returns an xarray DataArray with the correct dimensions.

    """

    datastore = init_datastore_example(datastore_name)

    for category in ["state", "forcing", "static"]:
        n_features = {}
        if category in ["state", "forcing"]:
            splits = ["train", "val", "test"]
        elif category == "static":
            # static data should be the same for all splits, so split
            # should be allowed to be None
            splits = ["train", "val", "test", None]
        else:
            raise NotImplementedError(category)

        for split in splits:
            expected_dims = ["grid_index", f"{category}_feature"]
            if category != "static":
                if not datastore.is_forecast:
                    expected_dims.append("time")
                else:
                    expected_dims += [
                        "analysis_time",
                        "elapsed_forecast_duration",
                    ]

            if datastore.is_ensemble and category == "state":
                # assume that only state variables change with ensemble members
                expected_dims.append("ensemble_member")

            # XXX: for now we only have a single attribute to get the shape of
            # the grid which uses the shape from the "state" category, maybe
            # this should change?

            da = datastore.get_dataarray(category=category, split=split)

            assert isinstance(da, xr.DataArray)
            assert set(da.dims) == set(expected_dims)
            if isinstance(datastore, BaseRegularGridDatastore):
                grid_shape = datastore.grid_shape_state
                assert da.grid_index.size == grid_shape.x * grid_shape.y

            n_features[split] = da[category + "_feature"].size

        # check that the number of features is the same for all splits
        assert n_features["train"] == n_features["val"] == n_features["test"]


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_boundary_mask(datastore_name):
    """Check that the `datastore.boundary_mask` property is implemented and
    that the returned object is an xarray DataArray with the correct shape."""
    datastore = init_datastore_example(datastore_name)
    da_mask = datastore.boundary_mask

    assert isinstance(da_mask, xr.DataArray)
    assert set(da_mask.dims) == {"grid_index"}
    assert da_mask.dtype == "int"
    assert set(da_mask.values) == {0, 1}
    assert da_mask.sum() > 0
    assert da_mask.sum() < da_mask.size

    if isinstance(datastore, BaseRegularGridDatastore):
        grid_shape = datastore.grid_shape_state
        assert datastore.boundary_mask.size == grid_shape.x * grid_shape.y


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_xy_extent(datastore_name):
    """Check that the `datastore.get_xy_extent` method is implemented and that
    the returned object is a tuple of the correct length."""
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip("Datastore does not implement `BaseCartesianDatastore`")

    extents = {}
    # get the extents for each category, and finally check they are all the same
    for category in ["state", "forcing", "static"]:
        extent = datastore.get_xy_extent(category)
        assert isinstance(extent, list)
        assert len(extent) == 4
        assert all(isinstance(e, (int, float)) for e in extent)
        extents[category] = extent

    # check that the extents are the same for all categories
    for category in ["forcing", "static"]:
        assert extents["state"] == extents[category]


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_xy(datastore_name):
    """Check that the `datastore.get_xy` method is implemented."""
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip("Datastore does not implement `BaseCartesianDatastore`")

    for category in ["state", "forcing", "static"]:
        xy_stacked = datastore.get_xy(category=category, stacked=True)
        xy_unstacked = datastore.get_xy(category=category, stacked=False)

        assert isinstance(xy_stacked, np.ndarray)
        assert isinstance(xy_unstacked, np.ndarray)

        nx, ny = datastore.grid_shape_state.x, datastore.grid_shape_state.y

        # for stacked=True, the shape should be (n_grid_points, 2)
        assert xy_stacked.ndim == 2
        assert xy_stacked.shape[0] == nx * ny
        assert xy_stacked.shape[1] == 2

        # for stacked=False, the shape should be (nx, ny, 2)
        assert xy_unstacked.ndim == 3
        assert xy_unstacked.shape[0] == nx
        assert xy_unstacked.shape[1] == ny
        assert xy_unstacked.shape[2] == 2


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_projection(datastore_name):
    """Check that the `datastore.coords_projection` property is implemented."""
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip("Datastore does not implement `BaseCartesianDatastore`")

    assert isinstance(datastore.coords_projection, ccrs.Projection)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def get_grid_shape_state(datastore_name):
    """Check that the `datastore.grid_shape_state` property is implemented."""
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip("Datastore does not implement `BaseCartesianDatastore`")

    grid_shape = datastore.grid_shape_state
    assert isinstance(grid_shape, tuple)
    assert len(grid_shape) == 2
    assert all(isinstance(e, int) for e in grid_shape)
    assert all(e > 0 for e in grid_shape)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
@pytest.mark.parametrize("category", ["state", "forcing", "static"])
def test_stacking_grid_coords(datastore_name, category):
    """Check that the `datastore.stack_grid_coords` method is implemented."""
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip("Datastore does not implement `BaseCartesianDatastore`")

    da_static = datastore.get_dataarray(category=category, split="train")

    da_static_unstacked = datastore.unstack_grid_coords(da_static).load()
    da_static_test = datastore.stack_grid_coords(da_static_unstacked)

    assert da_static.dims == da_static_test.dims
    xr.testing.assert_equal(da_static, da_static_test)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_dataarray_shapes(datastore_name):
    datastore = init_datastore_example(datastore_name)
    static_da = datastore.get_dataarray("static", split=None)
    static_da = datastore.stack_grid_coords(static_da)
    static_da = static_da.isel(static_feature=0)

    # Convert the unstacked grid coordinates and static data array to tensors
    unstacked_tensor = torch.tensor(
        datastore.unstack_grid_coords(static_da).to_numpy(), dtype=torch.float32
    ).squeeze()

    reshaped_tensor = (
        torch.tensor(static_da.to_numpy(), dtype=torch.float32)
        .reshape(datastore.grid_shape_state.x, datastore.grid_shape_state.y)
        .squeeze()
    )

    # Compute the difference
    diff = unstacked_tensor - reshaped_tensor

    # Check the shapes
    assert unstacked_tensor.shape == (
        datastore.grid_shape_state.x,
        datastore.grid_shape_state.y,
    )
    assert reshaped_tensor.shape == (
        datastore.grid_shape_state.x,
        datastore.grid_shape_state.y,
    )
    assert diff.shape == (
        datastore.grid_shape_state.x,
        datastore.grid_shape_state.y,
    )
    # assert diff == 0 with tolerance 1e-6
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-6)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_plot_example_from_datastore(datastore_name):
    """Check that the `plot_example_from_datastore` function is implemented."""
    datastore = init_datastore_example(datastore_name)
    fig = plot_example_from_datastore(
        category="static",
        datastore=datastore,
        col_dim="{category}_feature",
        split="train",
        standardize=True,
        selection={},
        index_selection={},
    )

    assert fig is not None
    assert fig.get_axes()


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
@pytest.mark.parametrize("category", ("state", "static"))
def test_get_standardized_da(datastore_name, category):
    """Check that dataarray is actually standardized when calling
    get_dataarray with standardize=True"""
    datastore = init_datastore_example(datastore_name)
    ds_stats = datastore.get_standardization_dataarray(category=category)

    mean = ds_stats[f"{category}_mean"]
    std = ds_stats[f"{category}_std"]

    non_standard_da = datastore.get_dataarray(
        category=category, split="train", standardize=False
    )
    standard_da = datastore.get_dataarray(
        category=category, split="train", standardize=True
    )

    assert np.allclose(standard_da, (non_standard_da - mean) / std, atol=1e-6)
