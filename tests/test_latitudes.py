# Third-party
import numpy as np

# Local
from .dummy_datastore import DummyDatastore


def test_get_latitudes_dummy_datastore():
    datastore = DummyDatastore()

    for category in ["state", "forcing", "static"]:
        latitudes = datastore.get_latitudes(category=category)

        assert isinstance(latitudes, np.ndarray)
        assert latitudes.ndim == 1
        assert latitudes.size == datastore.num_grid_points
        assert np.isfinite(latitudes).all()
        assert (latitudes >= -90.0).all()
        assert (latitudes <= 90.0).all()
