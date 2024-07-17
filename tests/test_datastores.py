# Third-party
import pytest

# First-party
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


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_datastore(datastore_name):
    DatastoreClass = DATASTORES[datastore_name]
    datastore = DatastoreClass(**EXAMPLES[datastore_name])

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
