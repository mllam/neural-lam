# Third-party
import numpy as np
import torch

# First-party
from neural_lam import metrics
from neural_lam.models.ar_model import ARModel
from tests.dummy_datastore import DummyDatastore


def test_crps_ens_single_member_shape_matches_multi_member_when_sum_vars_false():
    batch_size = 2
    n_grid_nodes = 4
    n_state_features = 3

    target = torch.randn(batch_size, n_grid_nodes, n_state_features)
    pred_single = torch.randn(batch_size, 1, n_grid_nodes, n_state_features)
    pred_multi = torch.cat((pred_single, pred_single + 0.25), dim=1)

    metric_single = metrics.crps_ens(
        pred_single,
        target,
        None,
        average_grid=True,
        sum_vars=False,
        ens_dim=1,
    )
    metric_multi = metrics.crps_ens(
        pred_multi,
        target,
        None,
        average_grid=True,
        sum_vars=False,
        ens_dim=1,
    )

    assert metric_single.shape == metric_multi.shape
    assert metric_single.shape == (batch_size, n_state_features)


class _MinimalARModel:
    def __init__(self, datastore):
        self._datastore = datastore


def test_create_dataarray_from_tensor_supports_ensemble_member_dimension():
    datastore = DummyDatastore(n_grid_points=100, n_timesteps=8)
    model = _MinimalARModel(datastore=datastore)

    create_da = ARModel._create_dataarray_from_tensor.__get__(
        model, _MinimalARModel
    )

    n_time = 3
    n_ens = 2
    n_grid_nodes = datastore.num_grid_points
    n_state_features = datastore.get_num_data_vars(category="state")

    tensor = torch.randn(n_time, n_grid_nodes, n_ens, n_state_features)
    time = torch.tensor([0, 1, 2], dtype=torch.int64)

    da = create_da(
        tensor=tensor,
        time=time,
        split="train",
        category="state",
    )

    assert da.dims == (
        "time",
        "grid_index",
        "ensemble_member",
        "state_feature",
    )
    assert da.sizes["time"] == n_time
    assert da.sizes["grid_index"] == n_grid_nodes
    assert da.sizes["ensemble_member"] == n_ens
    assert da.sizes["state_feature"] == n_state_features
    np.testing.assert_array_equal(
        da.coords["ensemble_member"].values,
        np.arange(n_ens),
    )
