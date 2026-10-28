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

In addition, datastores on a rectilinear grid (`is_on_rectilinear_grid` is
`True`) must have the following methods and attributes:
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
from datetime import timedelta
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

# First-party
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES, init_datastore
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.datastore.plot_example import plot_example_from_datastore
from tests.conftest import (
    DATASTORE_EXAMPLES_ROOT_PATH,
    DATASTORES_EXAMPLES,
    init_datastore_boundary_example,
    init_datastore_example,
)


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
    assert isinstance(step_length, timedelta)
    assert step_length.total_seconds() > 0


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_datastore_grid_xy(datastore_name):
    """Use the `datastore.get_xy` method to get the x, y coordinates of the
    dataset and check that the shape is correct against the `da
    tastore.grid_shape_state` property."""
    datastore = init_datastore_example(datastore_name)

    if not datastore.is_on_rectilinear_grid:
        pytest.skip(
            "Skip grid_shape_state test for non-rectilinear grid datastores"
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

            if (
                datastore.is_ensemble
                and category == "state"
                or category == "forcing"
                and getattr(datastore, "has_ensemble_forcing", False)
            ):
                expected_dims.append("ensemble_member")

            # XXX: for now we only have a single attribute to get the shape of
            # the grid which uses the shape from the "state" category, maybe
            # this should change?

            da = datastore.get_dataarray(category=category, split=split)

            assert isinstance(da, xr.DataArray)
            assert set(da.dims) == set(expected_dims)
            if datastore.is_on_rectilinear_grid:
                grid_shape = datastore.grid_shape_state
                assert da.grid_index.size == grid_shape.x * grid_shape.y

            n_features[split] = da[category + "_feature"].size

        # check that the number of features is the same for all splits
        assert n_features["train"] == n_features["val"] == n_features["test"]


@pytest.mark.requires_real_data
def test_mdp_state_only_optional_categories(tmp_path):
    """Test MDPDatastore accessors for absent optional categories.

    Builds the datastore from the DANRA example config with the `forcing`
    and `static` outputs and their inputs removed. For these optional
    categories, `get_dataarray` returns `None` and `get_vars_names`,
    `get_vars_units` and `get_vars_long_names` return empty lists, each
    emitting a `UserWarning`.
    """
    danra_config_path = (
        DATASTORE_EXAMPLES_ROOT_PATH
        / "mdp"
        / "danra_100m_winds"
        / "danra.datastore.yaml"
    )
    config = yaml.safe_load(danra_config_path.read_text())
    for category in ["forcing", "static"]:
        del config["output"]["variables"][category]
    config["inputs"] = {
        name: input_
        for name, input_ in config["inputs"].items()
        if input_["target_output_variable"] == "state"
    }
    config_path = tmp_path / "state_only.datastore.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.warns(UserWarning, match="no static data found"):
        datastore = MDPDatastore(config_path=config_path)

    assert len(datastore.get_vars_names("state")) > 0
    for category in ["forcing", "static"]:
        with pytest.warns(UserWarning, match=f"no {category} data found"):
            da = datastore.get_dataarray(category=category, split=None)
        assert da is None
        for get_vars in [
            datastore.get_vars_names,
            datastore.get_vars_units,
            datastore.get_vars_long_names,
        ]:
            with pytest.warns(UserWarning, match=f"no {category} data found"):
                assert get_vars(category=category) == []


@pytest.mark.requires_real_data
def test_npyfilesmeps_analysis_times_unique():
    """Each analysis time has one file per ensemble member, but must only
    appear once in the `analysis_time` coordinate."""
    datastore = init_datastore_example("npyfilesmeps")
    assert datastore.is_ensemble

    times = datastore._get_analysis_times(split="train")
    assert len(times) == len(set(times))
    assert times == sorted(times)

    da_state = datastore.get_dataarray(category="state", split="train")
    assert da_state.analysis_time.to_index().is_unique


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

    if datastore.is_on_rectilinear_grid:
        grid_shape = datastore.grid_shape_state
        assert datastore.boundary_mask.size == grid_shape.x * grid_shape.y


def test_init_datastore_rejects_n_boundary_points_for_non_mdp():
    """`n_boundary_points` is only supported for `MDPDatastore`."""
    with pytest.raises(ValueError, match="n_boundary_points"):
        init_datastore(
            datastore_kind="npyfilesmeps", config_path="", n_boundary_points=5
        )


@pytest.mark.requires_real_data
@pytest.mark.parametrize("n_boundary_points", [0, 1, 30])
def test_mdp_boundary_mask_n_boundary_points(n_boundary_points):
    """Check that `MDPDatastore.boundary_mask` marks exactly the outer
    `n_boundary_points` grid points along each edge as boundary, and none
    for `n_boundary_points=0`."""
    datastore = MDPDatastore(
        config_path=DATASTORES_EXAMPLES["mdp"],
        n_boundary_points=n_boundary_points,
    )
    grid_shape = datastore.grid_shape_state
    n_interior = (grid_shape.x - 2 * n_boundary_points) * (
        grid_shape.y - 2 * n_boundary_points
    )
    assert int(datastore.boundary_mask.sum()) == (
        grid_shape.x * grid_shape.y - n_interior
    )


def _write_mdp_grid_config(tmp_path, x_dim, y_dim, grid_index_dims):
    """Write a 10 x 8 grid zarr with dims (time, x_dim, y_dim) and an
    mllam-data-prep config stacking `grid_index` from `grid_index_dims`."""
    n_times = 24
    time = np.datetime64("2022-04-01T00:00") + np.arange(
        n_times
    ) * np.timedelta64(3, "h")
    rng = np.random.default_rng(0)
    xr.Dataset(
        {"t2m": (("time", x_dim, y_dim), rng.random((n_times, 10, 8)))},
        coords={
            "time": time,
            x_dim: np.linspace(0.0, 9.0, 10),
            y_dim: np.linspace(50.0, 57.0, 8),
        },
    ).to_zarr(tmp_path / "input.zarr", mode="w")

    config = {
        "schema_version": "v0.5.0",
        "dataset_version": "v0.1.0",
        "output": {
            "variables": {"state": ["time", "grid_index", "state_feature"]},
            "coord_ranges": {
                "time": {
                    "start": "2022-04-01T00:00",
                    "end": "2022-04-03T21:00",
                    "step": "PT3H",
                }
            },
            "splitting": {
                "dim": "time",
                "splits": {
                    "train": {
                        "start": "2022-04-01T00:00",
                        "end": "2022-04-02T00:00",
                        "compute_statistics": {
                            "ops": ["mean", "std", "diff_mean", "diff_std"],
                            "dims": ["grid_index", "time"],
                        },
                    },
                    "val": {
                        "start": "2022-04-02T03:00",
                        "end": "2022-04-03T00:00",
                    },
                    "test": {
                        "start": "2022-04-03T03:00",
                        "end": "2022-04-03T21:00",
                    },
                },
            },
        },
        "inputs": {
            "surface": {
                "path": str(tmp_path / "input.zarr"),
                "dims": ["time", x_dim, y_dim],
                "variables": ["t2m"],
                "dim_mapping": {
                    "time": {"method": "rename", "dim": "time"},
                    "grid_index": {"method": "stack", "dims": grid_index_dims},
                    "state_feature": {
                        "method": "stack_variables_by_var_name",
                        "name_format": "{var_name}",
                    },
                },
                "target_output_variable": "state",
            }
        },
        "extra": {"projection": {"class_name": "PlateCarree", "kwargs": {}}},
    }
    config_path = tmp_path / "grid.datastore.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@pytest.mark.requires_real_data
def test_mdp_geographic_spatial_coordinates(tmp_path):
    """`MDPDatastore` must work when `grid_index` is stacked from dimensions
    not named x/y, e.g. a regular longitude/latitude grid."""
    config_path = _write_mdp_grid_config(
        tmp_path, "longitude", "latitude", ["longitude", "latitude"]
    )
    datastore = MDPDatastore(config_path=config_path, n_boundary_points=2)

    assert list(datastore.spatial_coordinates) == ["longitude", "latitude"]
    assert datastore.grid_shape_state.x == 10
    assert datastore.grid_shape_state.y == 8
    assert datastore.get_dataarray("state", split="train") is not None
    # 10 x 8 grid with a 2-point frame leaves a 6 x 4 interior
    assert int(datastore.boundary_mask.sum()) == 10 * 8 - 6 * 4


@pytest.mark.requires_real_data
@pytest.mark.parametrize(
    "x_dim, y_dim", [("x", "y"), ("longitude", "latitude")]
)
def test_mdp_reversed_spatial_coordinates_raises(tmp_path, x_dim, y_dim):
    """Stacking `grid_index` y-first would silently swap x and y in
    `grid_shape_state`, `get_xy` and `get_xy_extent`, so it must raise."""
    config_path = _write_mdp_grid_config(tmp_path, x_dim, y_dim, [y_dim, x_dim])
    with pytest.raises(ValueError, match="x-like dimension must come first"):
        MDPDatastore(config_path=config_path)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_xy_extent(datastore_name):
    """Check that the `datastore.get_xy_extent` method is implemented and that
    the returned object is a tuple of the correct length."""
    datastore = init_datastore_example(datastore_name)

    if not datastore.is_on_rectilinear_grid:
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

    if not datastore.is_on_rectilinear_grid:
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
        def test_get_lat_lon(datastore_name):
            """Check that the `datastore.get_lat_lon` helper returns valid
            shapes and values."""
            datastore = init_datastore_example(datastore_name)

            if not datastore.is_on_rectilinear_grid:
                pytest.skip(
                    "Datastore does not implement `BaseCartesianDatastore`"
                )

            nx, ny = datastore.grid_shape_state.x, datastore.grid_shape_state.y

            lonlat = datastore.get_lat_lon(category="state")
            lonlat_grid = lonlat.reshape(nx, ny, 2)

            assert lonlat.shape == (nx * ny, 2)
            assert lonlat_grid.shape == (nx, ny, 2)
            assert np.isfinite(lonlat).all()
            assert np.isfinite(lonlat_grid).all()

            lon = lonlat[:, 0]
            lat = lonlat[:, 1]

            assert np.all((lat >= -90.0) & (lat <= 90.0))
            lon_in_180 = (lon >= -180.0) & (lon <= 180.0)
            lon_in_360 = (lon >= 0.0) & (lon <= 360.0)
            assert np.all(lon_in_180 | lon_in_360)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_projection(datastore_name):
    """Check that the `datastore.coords_projection` property is implemented."""
    datastore = init_datastore_example(datastore_name)

    if not datastore.is_on_rectilinear_grid:
        pytest.skip("Datastore does not implement `BaseCartesianDatastore`")

    assert isinstance(datastore.coords_projection, ccrs.Projection)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_grid_shape_state(datastore_name):
    """Check that the example datastores are on a rectilinear grid whose
    `grid_shape_state` accounts for every grid point."""
    datastore = init_datastore_example(datastore_name)
    assert datastore.is_on_rectilinear_grid

    grid_shape = datastore.grid_shape_state
    assert grid_shape.x > 0
    assert grid_shape.y > 0
    assert grid_shape.x * grid_shape.y == datastore.num_grid_points


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
@pytest.mark.parametrize("category", ["state", "forcing", "static"])
def test_stacking_grid_coords(datastore_name, category):
    """Check that the `datastore.stack_grid_coords` method is implemented."""
    datastore = init_datastore_example(datastore_name)

    if not datastore.is_on_rectilinear_grid:
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


@pytest.mark.requires_real_data
def test_cropped_boundary_datastore_is_not_rectilinear_grid():
    """A domain-cropped datastore is not on a rectilinear grid, so unstacking
    and graph creation refuse it."""
    datastore_boundary = init_datastore_boundary_example("mdp")

    xy = datastore_boundary.get_xy(category="forcing", stacked=True)
    assert xy.shape == (datastore_boundary.num_grid_points, 2)
    n_x, n_y = np.unique(xy[:, 0]).size, np.unique(xy[:, 1]).size
    assert datastore_boundary.num_grid_points < n_x * n_y
    assert not datastore_boundary.is_on_rectilinear_grid

    with pytest.raises(NotImplementedError, match="complete 2D grid"):
        _ = datastore_boundary.grid_shape_state

    da_forcing = datastore_boundary.get_dataarray(
        category="forcing", split="train"
    )
    with pytest.raises(NotImplementedError, match="complete 2D grid"):
        datastore_boundary.unstack_grid_coords(da_forcing)

    with pytest.raises(NotImplementedError, match="rectilinear grid"):
        create_graph_from_datastore(
            datastore=datastore_boundary,
            output_root_path=str(datastore_boundary.root_path / "graph"),
        )


@pytest.mark.requires_real_data
def test_boundary_mask_on_boundary_datastore_raises():
    """`boundary_mask` is not defined without `state` data."""
    datastore_boundary = init_datastore_boundary_example("mdp")

    with pytest.raises(NotImplementedError, match="with `state` data"):
        datastore_boundary.boundary_mask


@pytest.mark.requires_real_data
def test_boundary_datastore_state_metadata_accessors_return_empty():
    """The state metadata of a datastore without `state` data is empty."""
    datastore_boundary = init_datastore_boundary_example("mdp")

    assert datastore_boundary.get_vars_names(category="state") == []
    assert datastore_boundary.get_vars_long_names(category="state") == []
    assert datastore_boundary.get_vars_units(category="state") == []


@pytest.mark.requires_real_data
def test_symlinked_config_resolves_to_the_real_directory():
    """A config reached through a symlink, like the DANRA config of the
    boundary example, uses the link target as its root."""
    linked = (
        Path(__file__).parent
        / "datastore_examples"
        / "mdp"
        / "era5_1000hPa_danra_100m_winds"
        / "danra.datastore.yaml"
    )
    assert linked.is_symlink()

    datastore = init_datastore(datastore_kind="mdp", config_path=linked)

    assert datastore.root_path == linked.resolve().parent
    assert datastore.root_path.name == "danra_100m_winds"
