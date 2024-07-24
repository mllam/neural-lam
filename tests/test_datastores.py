"""List of methods and attributes that should be implemented in a subclass of
`BaseCartesianDatastore` (these are all decorated with `@abc.abstractmethod`):

- [x] `grid_shape_state` (property): Shape of the grid for the state variables.
- [x] `get_xy` (method): Return the x, y coordinates of the dataset.
- [x] `coords_projection` (property): Projection object for the coordinates.
- [x] `get_vars_units` (method): Get the units of the variables in the given category.
- [x] `get_vars_names` (method): Get the names of the variables in the given category.
- [x] `get_num_data_vars` (method): Get the number of data variables in the
      given category.
- [x] `get_normalization_dataarray` (method): Return the normalization
      dataarray for the given category.
- [x] `get_dataarray` (method): Return the processed data (as a single
      `xr.DataArray`) for the given category and test/train/val-split.
- [ ] `boundary_mask` (property): Return the boundary mask for the dataset,
      with spatial dimensions stacked.

In addition BaseCartesianDatastore must have the following methods and attributes:
- [ ] `get_xy_extent` (method): Return the extent of the x, y coordinates for a
        given category of data.
- [ ] `get_xy` (method): Return the x, y coordinates of the dataset.
- [ ] `coords_projection` (property): Projection object for the coordinates.
- [ ] `grid_shape_state` (property): Shape of the grid for the state variables.
"""

# Third-party
import cartopy.crs as ccrs
import pytest
import xarray as xr

# First-party
from neural_lam.datastore.base import BaseCartesianDatastore
from neural_lam.datastore.mllam import MLLAMDatastore
from neural_lam.datastore.multizarr import MultiZarrDatastore
from neural_lam.datastore.npyfiles import NumpyFilesDatastore

DATASTORES = dict(
    multizarr=MultiZarrDatastore,
    mllam=MLLAMDatastore,
    npyfiles=NumpyFilesDatastore,
)


EXAMPLES = dict(
    multizarr=dict(
        config_path="tests/datastore_configs/multizarr/data_config.yaml"
    ),
    mllam=dict(config_path="tests/datastore_configs/mllam/example.danra.yaml"),
    npyfiles=dict(root_path="tests/datastore_configs/npy"),
)


def _init_datastore(datastore_name):
    DatastoreClass = DATASTORES[datastore_name]
    return DatastoreClass(**EXAMPLES[datastore_name])


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_datastore_grid_xy(datastore_name):
    """Use the `datastore.get_xy` method to get the x, y coordinates of the
    dataset and check that the shape is correct against the
    `datastore.grid_shape_state` property."""
    datastore = _init_datastore(datastore_name)

    # check the shapes of the xy grid
    grid_shape = datastore.grid_shape_state
    nx, ny = grid_shape.x, grid_shape.y
    for stacked in [True, False]:
        xy = datastore.get_xy("static", stacked=stacked)
        """
            - `stacked==True`: shape `(2, n_grid_points)` where n_grid_points=N_x*N_y.
            - `stacked==False`: shape `(2, N_y, N_x)`
        """
        if stacked:
            assert xy.shape == (2, nx * ny)
        else:
            assert xy.shape == (2, ny, nx)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_projection(datastore_name):
    """Check that the `datastore.coords_projection` property is implemented."""
    datastore = _init_datastore(datastore_name)

    assert isinstance(datastore.coords_projection, ccrs.Projection)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_vars(datastore_name):
    """Check that results of.

    - `datastore.get_vars_units`
    - `datastore.get_vars_names`
    - `datastore.get_num_data_vars`

    are consistent (as in the number of variables are the same) and that the
    return types of each are correct.
    """
    datastore = _init_datastore(datastore_name)

    for category in ["state", "forcing", "static"]:
        units = datastore.get_vars_units(category)
        names = datastore.get_vars_names(category)
        num_vars = datastore.get_num_data_vars(category)

        assert len(units) == len(names) == num_vars
        assert isinstance(units, list)
        assert isinstance(names, list)
        assert isinstance(num_vars, int)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_normalization_dataarray(datastore_name):
    """Check that the `datastore.get_normalization_dataarray` method is
    implemented."""
    datastore = _init_datastore(datastore_name)

    for category in ["state", "forcing", "static"]:
        ds_stats = datastore.get_normalization_dataarray(category=category)

        # check that the returned object is an xarray DataArray
        # and that it has the correct variables
        assert isinstance(ds_stats, xr.Dataset)

        if category == "state":
            ops = ["mean", "std", "diff_mean", "diff_std"]
        elif category == "forcing":
            ops = ["mean", "std"]
        elif category == "static":
            ops = []
        else:
            raise NotImplementedError(category)

        for op in ops:
            var_name = f"{category}_{op}"
            assert var_name in ds_stats.data_vars


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_get_dataarray(datastore_name):
    """Check that the `datastore.get_dataarray` method is implemented.

    And that it returns an xarray DataArray with the correct dimensions.
    """

    datastore = _init_datastore(datastore_name)

    for category in ["state", "forcing", "static"]:
        for split in ["train", "val", "test"]:
            expected_dims = ["grid_index", f"{category}_feature"]
            if category != "static":
                if not datastore.is_forecast:
                    expected_dims.append("time")
                else:
                    expected_dims += [
                        "analysis_time",
                        "elapsed_forecast_duration",
                    ]

            # XXX: for now we only have a single attribute to get the shape of
            # the grid which uses the shape from the "state" category, maybe
            # this should change?
            grid_shape = datastore.grid_shape_state

            da = datastore.get_dataarray(category=category, split=split)

            assert isinstance(da, xr.DataArray)
            assert set(da.dims) == set(expected_dims)
            if isinstance(datastore, BaseCartesianDatastore):
                assert da.grid_index.size == grid_shape.x * grid_shape.y


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_boundary_mask(datastore_name):
    """Check that the `datastore.boundary_mask` property is implemented and
    that the returned object is an xarray DataArray with the correct shape."""
    datastore = _init_datastore(datastore_name)
    da_mask = datastore.boundary_mask

    assert isinstance(da_mask, xr.DataArray)
    assert set(da_mask.dims) == {"grid_index"}
    assert da_mask.dtype == "int"
    assert set(da_mask.values) == {0, 1}
    assert da_mask.sum() > 0
    assert da_mask.sum() < da_mask.size

    if isinstance(datastore, BaseCartesianDatastore):
        grid_shape = datastore.grid_shape_state
        assert datastore.boundary_mask.size == grid_shape.x * grid_shape.y
