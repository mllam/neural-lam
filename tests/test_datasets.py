# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore, EnsembleDummyDatastore


class ForecastArrayDatastore(DummyDatastore):
    is_forecast = True

    def __init__(self, da_state, da_forcing):
        super().__init__(n_grid_points=1, n_timesteps=1)
        self._state_da = da_state
        self._forcing_da = da_forcing
        self.is_ensemble = "ensemble_member" in da_state.dims
        self.has_ensemble_forcing = (
            da_forcing is not None and "ensemble_member" in da_forcing.dims
        )

    def get_dataarray(self, category, split, **kwargs):
        if category == "state":
            return self._state_da
        if category == "forcing":
            return self._forcing_da
        return super().get_dataarray(category, split, **kwargs)


def make_forecast_dataset(
    da_state,
    da_forcing,
    *,
    ar_steps,
    num_past_forcing_steps,
    num_future_forcing_steps,
):
    datastore = ForecastArrayDatastore(da_state=da_state, da_forcing=da_forcing)
    return WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=ar_steps,
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        standardize=False,
    )


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_dataset_item_shapes(datastore_name):
    """Check that the `datastore.get_dataarray` method is implemented.

    Validate the shapes of the tensors match between the different
    components of the training sample.

    init_states: (2, N_grid, d_features)
    target_states: (ar_steps, N_grid, d_features)
    forcing: (ar_steps, N_grid, d_windowed_forcing) # batch_times: (ar_steps,)

    """
    datastore = init_datastore_example(datastore_name)
    N_gridpoints = datastore.num_grid_points

    N_pred_steps = 4
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=N_pred_steps,
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
    )

    item = dataset[0]

    # unpack the item, this is the current return signature for
    # WeatherDataset.__getitem__
    init_states, target_states, forcing, target_times = item

    # initial states
    assert init_states.ndim == 3
    assert init_states.shape[0] == 2  # two time steps go into the input
    assert init_states.shape[1] == N_gridpoints
    assert init_states.shape[2] == datastore.get_num_data_vars("state")

    # output states
    assert target_states.ndim == 3
    assert target_states.shape[0] == N_pred_steps
    assert target_states.shape[1] == N_gridpoints
    assert target_states.shape[2] == datastore.get_num_data_vars("state")

    # forcing
    assert forcing.ndim == 3
    assert forcing.shape[0] == N_pred_steps
    assert forcing.shape[1] == N_gridpoints
    assert forcing.shape[2] == datastore.get_num_data_vars("forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )

    # batch times
    assert target_times.ndim == 1
    assert target_times.shape[0] == N_pred_steps

    # try to get the last item of the dataset to ensure slicing and stacking
    # operations are working as expected and are consistent with the dataset
    # length
    dataset[len(dataset) - 1]


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_dataset_item_create_dataarray_from_tensor(datastore_name):
    datastore = init_datastore_example(datastore_name)

    N_pred_steps = 4
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=N_pred_steps,
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
    )

    idx = 0

    # unpack the item, this is the current return signature for
    # WeatherDataset.__getitem__
    _, target_states, _, target_times_arr = dataset[idx]
    _, da_target_true, _, da_target_times_true = dataset._build_item_dataarrays(
        idx=idx
    )

    target_times = np.array(target_times_arr, dtype="datetime64[ns]")
    np.testing.assert_equal(target_times, da_target_times_true.values)

    da_target = dataset.create_dataarray_from_tensor(
        tensor=target_states, category="state", time=target_times
    )

    # conversion to torch.float32 may lead to loss of precision
    np.testing.assert_allclose(
        da_target.values, da_target_true.values, rtol=1e-6
    )
    assert da_target.dims == da_target_true.dims
    for dim in da_target.dims:
        np.testing.assert_equal(
            da_target[dim].values, da_target_true[dim].values
        )

    if isinstance(datastore, BaseRegularGridDatastore):
        # test unstacking the grid coordinates
        da_target_unstacked = datastore.unstack_grid_coords(da_target)
        assert all(
            coord_name in da_target_unstacked.coords
            for coord_name in ["x", "y"]
        )

    # check construction of a single time
    da_target_single = dataset.create_dataarray_from_tensor(
        tensor=target_states[0], category="state", time=target_times[0]
    )

    # check that the content is the same
    # conversion to torch.float32 may lead to loss of precision
    np.testing.assert_allclose(
        da_target_single.values, da_target_true[0].values, rtol=1e-6
    )
    assert da_target_single.dims == da_target_true[0].dims
    for dim in da_target_single.dims:
        np.testing.assert_equal(
            da_target_single[dim].values, da_target_true[0][dim].values
        )

    if isinstance(datastore, BaseRegularGridDatastore):
        # test unstacking the grid coordinates
        da_target_single_unstacked = datastore.unstack_grid_coords(
            da_target_single
        )
        assert all(
            coord_name in da_target_single_unstacked.coords
            for coord_name in ["x", "y"]
        )


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_single_batch(datastore_name, split):
    """Check that the `datastore.get_dataarray` method is implemented.

    And that it returns an xarray DataArray with the correct dimensions.

    """
    datastore = init_datastore_example(datastore_name)

    device_name = (
        torch.device("cuda") if torch.cuda.is_available() else "cpu"
    )  # noqa

    graph_name = "1level"

    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        graph = graph_name
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 2
        mesh_aggr = "sum"
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1
        val_steps_to_log = [1, 3]
        ar_steps_eval = 5

    args = ModelArgs()

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    def _create_graph():
        if not graph_dir_path.exists():
            create_graph_from_datastore(
                datastore=datastore,
                output_root_path=str(graph_dir_path),
                n_max_levels=1,
            )

    if not isinstance(datastore, BaseRegularGridDatastore):
        with pytest.raises(NotImplementedError):
            _create_graph()
        pytest.skip("Skipping on model-run on non-regular grid datastores")

    _create_graph()

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    dataset = WeatherDataset(datastore=datastore, split=split, ar_steps=2)

    model = GraphLAM(args=args, datastore=datastore, config=config)  # noqa

    model_device = model.to(device_name)
    data_loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(data_loader))
    batch_device = [part.to(device_name) for part in batch]
    model_device.common_step(batch_device)
    model_device.training_step(batch_device)


@pytest.mark.parametrize(
    "dataset_config",
    [
        {"past": 0, "future": 0, "ar_steps": 1, "exp_len_reduction": 2},
        {"past": 2, "future": 0, "ar_steps": 1, "exp_len_reduction": 2},
        {"past": 0, "future": 2, "ar_steps": 1, "exp_len_reduction": 4},
        {"past": 4, "future": 0, "ar_steps": 1, "exp_len_reduction": 4},
        {"past": 0, "future": 0, "ar_steps": 5, "exp_len_reduction": 6},
        {"past": 3, "future": 3, "ar_steps": 2, "exp_len_reduction": 7},
    ],
)
def test_dataset_length(dataset_config):
    """Check that correct number of samples can be extracted from the dataset,
    given a specific configuration of forcing windowing and ar_steps.
    """
    # Use dummy datastore of length 10 here, only want to test slicing
    # in dataset class
    ds_len = 10
    datastore = DummyDatastore(n_timesteps=ds_len)

    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=dataset_config["ar_steps"],
        num_past_forcing_steps=dataset_config["past"],
        num_future_forcing_steps=dataset_config["future"],
    )

    # We expect dataset to contain this many samples
    expected_len = ds_len - dataset_config["exp_len_reduction"]

    # Check that datast has correct length
    assert len(dataset) == expected_len

    # Check that we can actually get last and first sample
    dataset[0]
    dataset[expected_len - 1]


def test_ensemble_len_scales_with_default_all_members():
    datastore = EnsembleDummyDatastore(
        is_forecast=False,
        forcing_has_ensemble=False,
        n_ensemble_members=3,
        n_timesteps=10,
    )

    dataset_all = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        standardize=False,
    )

    dataset_single = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        load_single_member=True,
        standardize=False,
    )

    assert len(dataset_all) == len(dataset_single) * 3


def test_expected_dim_order_handles_optional_ensemble_forcing():
    datastore_with_ensemble_forcing = EnsembleDummyDatastore(
        is_forecast=False,
        forcing_has_ensemble=True,
        n_ensemble_members=3,
        n_timesteps=10,
    )

    datastore_without_ensemble_forcing = EnsembleDummyDatastore(
        is_forecast=False,
        forcing_has_ensemble=False,
        n_ensemble_members=3,
        n_timesteps=10,
    )

    assert datastore_with_ensemble_forcing.is_ensemble is True
    assert datastore_with_ensemble_forcing.has_ensemble_forcing is True
    assert datastore_without_ensemble_forcing.is_ensemble is True
    assert datastore_without_ensemble_forcing.has_ensemble_forcing is False

    assert datastore_with_ensemble_forcing.expected_dim_order(
        category="forcing"
    ) == ("time", "ensemble_member", "grid_index", "forcing_feature")
    assert datastore_without_ensemble_forcing.expected_dim_order(
        category="forcing"
    ) == ("time", "grid_index", "forcing_feature")
    assert datastore_with_ensemble_forcing.expected_dim_order(
        category="static"
    ) == ("grid_index", "static_feature")


def test_ensemble_index_mapping_is_time_major():
    datastore = EnsembleDummyDatastore(
        is_forecast=False,
        forcing_has_ensemble=False,
        n_ensemble_members=3,
        n_timesteps=10,
    )
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        load_single_member=False,
        standardize=False,
    )

    init_states_0, _, _, target_times_0 = dataset[0]
    init_states_1, _, _, target_times_1 = dataset[1]

    # Adjacent flat indices correspond to same sample_idx and different member.
    assert torch.equal(target_times_0, target_times_1)
    assert not torch.equal(init_states_0, init_states_1)


def test_ensemble_forcing_uses_same_member_when_available():
    datastore = EnsembleDummyDatastore(
        is_forecast=False,
        forcing_has_ensemble=True,
        n_ensemble_members=3,
        n_timesteps=10,
    )
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        load_single_member=False,
        standardize=False,
    )

    _, _, forcing_0, target_times_0 = dataset[0]
    _, _, forcing_1, target_times_1 = dataset[1]

    assert torch.equal(target_times_0, target_times_1)
    assert not torch.equal(forcing_0, forcing_1)


def test_ensemble_forcing_without_member_dim_is_shared():
    datastore = EnsembleDummyDatastore(
        is_forecast=False,
        forcing_has_ensemble=False,
        n_ensemble_members=3,
        n_timesteps=10,
    )
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        load_single_member=False,
        standardize=False,
    )

    init_states_0, _, forcing_0, target_times_0 = dataset[0]
    init_states_1, _, forcing_1, target_times_1 = dataset[1]

    assert torch.equal(target_times_0, target_times_1)
    assert not torch.equal(init_states_0, init_states_1)
    assert torch.equal(forcing_0, forcing_1)


def test_forecast_ensemble_len_scales_with_default_all_members():
    datastore = EnsembleDummyDatastore(
        is_forecast=True,
        forcing_has_ensemble=True,
        n_ensemble_members=3,
        n_analysis_times=4,
        n_forecast_steps=6,
    )

    dataset_all = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        standardize=False,
    )

    with pytest.warns(UserWarning, match="only using first ensemble member"):
        dataset_single = WeatherDataset(
            datastore=datastore,
            split="train",
            ar_steps=2,
            num_past_forcing_steps=1,
            num_future_forcing_steps=1,
            load_single_member=True,
            standardize=False,
        )

    assert len(dataset_all) == len(dataset_single) * 3


def test_standardization_with_zero_std():
    """Regression test for https://github.com/mllam/neural-lam/issues/136

    When all values of a field are identical (std = 0), WeatherDataset
    must not produce NaN via division-by-zero during standardization.
    """
    # Third-party
    import xarray as xr

    std_da = xr.DataArray(
        np.array([0.0, 1.0, 2.0], dtype=np.float32), dims=["feature"]
    )

    dataset = WeatherDataset.__new__(WeatherDataset)
    result = dataset._compute_std_safe(std_da, "state")

    eps = np.finfo(std_da.dtype).eps

    assert (
        float(result[0]) == eps
    ), "Zero std was not clamped to machine epsilon"
    assert float(result[1]) == 1.0
    assert float(result[2]) == 2.0
    assert not np.isnan(
        result.values
    ).any(), "NaN found after _compute_std_safe"


def test_dataset_out_of_bounds_indexing_raises():
    """Ensure out-of-range indexing fails instead of returning bad samples."""
    datastore = DummyDatastore(n_grid_points=4, n_timesteps=10)
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )

    # In-bounds indices work, including Python-style negative indexing.
    dataset[0]
    dataset[len(dataset) - 1]
    dataset[-1]

    # Out-of-bounds indices must fail explicitly.
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(IndexError):
        dataset[len(dataset) + 1]
    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]


def test_negative_indexing_does_not_call_len_in_getitem():
    class LenBombDataset(WeatherDataset):
        def __len__(self):
            raise AssertionError("__getitem__ should use cached dataset length")

    datastore = DummyDatastore(n_grid_points=4, n_timesteps=10)
    dataset = LenBombDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )

    dataset[-1]


def test_forecast_len_raises_when_forcing_horizon_too_short():
    analysis_time = np.array(
        ["2021-01-01T00:00:00", "2021-01-01T01:00:00"],
        dtype="datetime64[ns]",
    )
    elapsed = np.arange(5, dtype="timedelta64[h]").astype("timedelta64[ns]")

    da_state = xr.DataArray(
        np.zeros((2, 5, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "state_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": elapsed,
            "grid_index": [0],
            "state_feature": ["state_feat_0"],
        },
    )
    da_forcing = xr.DataArray(
        np.zeros((2, 5, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "forcing_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": elapsed,
            "grid_index": [0],
            "forcing_feature": ["forcing_feat_0"],
        },
    )

    with pytest.raises(
        ValueError,
        match="forecast lead times must match|forcing forecast steps",
    ):
        make_forecast_dataset(
            da_state,
            da_forcing,
            ar_steps=2,
            num_past_forcing_steps=1,
            num_future_forcing_steps=2,
        )


def test_forecast_len_raises_when_state_horizon_too_short_for_past_forcing():
    analysis_time = np.array(
        ["2021-01-01T00:00:00", "2021-01-01T01:00:00"],
        dtype="datetime64[ns]",
    )
    elapsed = np.arange(4, dtype="timedelta64[h]").astype("timedelta64[ns]")

    da_state = xr.DataArray(
        np.zeros((2, 4, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "state_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": elapsed,
            "grid_index": [0],
            "state_feature": ["state_feat_0"],
        },
    )

    with pytest.raises(ValueError, match="initial and target states"):
        make_forecast_dataset(
            da_state,
            None,
            ar_steps=1,
            num_past_forcing_steps=4,
            num_future_forcing_steps=0,
        )


def test_forecast_len_accepts_exact_state_horizon_for_past_forcing():
    analysis_time = np.array(
        ["2021-01-01T00:00:00", "2021-01-01T01:00:00"],
        dtype="datetime64[ns]",
    )
    elapsed = np.arange(5, dtype="timedelta64[h]").astype("timedelta64[ns]")

    da_state = xr.DataArray(
        np.zeros((2, 5, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "state_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": elapsed,
            "grid_index": [0],
            "state_feature": ["state_feat_0"],
        },
    )

    dataset = make_forecast_dataset(
        da_state,
        None,
        ar_steps=1,
        num_past_forcing_steps=4,
        num_future_forcing_steps=0,
    )
    assert len(dataset) == 2


def test_forecast_len_accepts_exact_forcing_horizon():
    analysis_time = np.array(
        ["2021-01-01T00:00:00", "2021-01-01T01:00:00"],
        dtype="datetime64[ns]",
    )
    elapsed = np.arange(6, dtype="timedelta64[h]").astype("timedelta64[ns]")

    da_state = xr.DataArray(
        np.zeros((2, 6, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "state_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": elapsed,
            "grid_index": [0],
            "state_feature": ["state_feat_0"],
        },
    )
    da_forcing = xr.DataArray(
        np.zeros((2, 6, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "forcing_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": elapsed,
            "grid_index": [0],
            "forcing_feature": ["forcing_feat_0"],
        },
    )

    dataset = make_forecast_dataset(
        da_state,
        da_forcing,
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=2,
    )
    assert len(dataset) == 2


def test_forecast_len_accepts_longer_forcing_horizon_with_matching_prefix():
    analysis_time = np.array(
        ["2021-01-01T00:00:00", "2021-01-01T01:00:00"],
        dtype="datetime64[ns]",
    )
    state_elapsed = np.arange(4, dtype="timedelta64[h]").astype(
        "timedelta64[ns]"
    )
    forcing_elapsed = np.arange(6, dtype="timedelta64[h]").astype(
        "timedelta64[ns]"
    )

    da_state = xr.DataArray(
        np.zeros((2, 4, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "state_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": state_elapsed,
            "grid_index": [0],
            "state_feature": ["state_feat_0"],
        },
    )
    da_forcing = xr.DataArray(
        np.zeros((2, 6, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "forcing_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": forcing_elapsed,
            "grid_index": [0],
            "forcing_feature": ["forcing_feat_0"],
        },
    )

    dataset = make_forecast_dataset(
        da_state,
        da_forcing,
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=2,
    )

    assert len(dataset) == 2


def test_forecast_len_raises_when_forcing_shorter_than_state_horizon():
    analysis_time = np.array(
        ["2021-01-01T00:00:00", "2021-01-01T01:00:00"],
        dtype="datetime64[ns]",
    )
    state_elapsed = np.arange(6, dtype="timedelta64[h]").astype(
        "timedelta64[ns]"
    )
    forcing_elapsed = np.arange(5, dtype="timedelta64[h]").astype(
        "timedelta64[ns]"
    )

    da_state = xr.DataArray(
        np.zeros((2, 6, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "state_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": state_elapsed,
            "grid_index": [0],
            "state_feature": ["state_feat_0"],
        },
    )
    da_forcing = xr.DataArray(
        np.zeros((2, 5, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "forcing_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": forcing_elapsed,
            "grid_index": [0],
            "forcing_feature": ["forcing_feat_0"],
        },
    )

    with pytest.raises(
        ValueError,
        match="forecast lead times must match|forcing forecast steps",
    ):
        make_forecast_dataset(
            da_state,
            da_forcing,
            ar_steps=2,
            num_past_forcing_steps=1,
            num_future_forcing_steps=2,
        )


def test_forecast_len_raises_when_analysis_times_do_not_match():
    state_analysis_time = np.array(
        ["2021-01-01T00:00:00", "2021-01-01T01:00:00"],
        dtype="datetime64[ns]",
    )
    forcing_analysis_time = np.array(
        ["2021-01-01T00:00:00"],
        dtype="datetime64[ns]",
    )
    elapsed = np.arange(5, dtype="timedelta64[h]").astype("timedelta64[ns]")

    da_state = xr.DataArray(
        np.zeros((2, 5, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "state_feature",
        ),
        coords={
            "analysis_time": state_analysis_time,
            "elapsed_forecast_duration": elapsed,
            "grid_index": [0],
            "state_feature": ["state_feat_0"],
        },
    )
    da_forcing = xr.DataArray(
        np.zeros((1, 5, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "forcing_feature",
        ),
        coords={
            "analysis_time": forcing_analysis_time,
            "elapsed_forecast_duration": elapsed,
            "grid_index": [0],
            "forcing_feature": ["forcing_feat_0"],
        },
    )

    with pytest.raises(ValueError, match="analysis times must match"):
        make_forecast_dataset(
            da_state,
            da_forcing,
            ar_steps=2,
            num_past_forcing_steps=1,
            num_future_forcing_steps=1,
        )


def test_forecast_len_raises_when_forecast_lead_times_do_not_match():
    analysis_time = np.array(
        ["2021-01-01T00:00:00", "2021-01-01T01:00:00"],
        dtype="datetime64[ns]",
    )
    state_elapsed = np.arange(5, dtype="timedelta64[h]").astype(
        "timedelta64[ns]"
    )
    forcing_elapsed = np.array([0, 2, 4, 6, 8], dtype="timedelta64[h]").astype(
        "timedelta64[ns]"
    )

    da_state = xr.DataArray(
        np.zeros((2, 5, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "state_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": state_elapsed,
            "grid_index": [0],
            "state_feature": ["state_feat_0"],
        },
    )
    da_forcing = xr.DataArray(
        np.zeros((2, 5, 1, 1), dtype=np.float32),
        dims=(
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            "forcing_feature",
        ),
        coords={
            "analysis_time": analysis_time,
            "elapsed_forecast_duration": forcing_elapsed,
            "grid_index": [0],
            "forcing_feature": ["forcing_feat_0"],
        },
    )

    with pytest.raises(ValueError, match="forecast lead times must match"):
        make_forecast_dataset(
            da_state,
            da_forcing,
            ar_steps=2,
            num_past_forcing_steps=1,
            num_future_forcing_steps=1,
        )


def test_weather_dataset_forecast_empty_split_raises_value_error():
    """Empty forecast splits should raise the intended user-facing error."""
    # Third-party
    import xarray as xr

    class EmptyForecastDatastore(DummyDatastore):
        is_forecast = True

        def get_dataarray(self, category, split, **kwargs):
            if category == "state":
                return xr.DataArray(
                    np.zeros((0, 3, 1, 1), dtype=np.float32),
                    dims=(
                        "analysis_time",
                        "elapsed_forecast_duration",
                        "grid_index",
                        "state_feature",
                    ),
                    coords={
                        "analysis_time": np.array([], dtype="datetime64[ns]"),
                        "elapsed_forecast_duration": np.arange(
                            3, dtype="timedelta64[h]"
                        ).astype("timedelta64[ns]"),
                        "grid_index": [0],
                        "state_feature": ["state_feat_0"],
                    },
                )
            if category == "forcing":
                return None
            return super().get_dataarray(
                category=category, split=split, **kwargs
            )

    datastore = EmptyForecastDatastore(n_grid_points=4, n_timesteps=10)

    with pytest.raises(ValueError, match="0 total time steps"):
        WeatherDataset(
            datastore=datastore,
            split="train",
            ar_steps=1,
            num_past_forcing_steps=1,
            num_future_forcing_steps=0,
            standardize=False,
        )


def test_analysis_len_limited_by_shorter_forcing_horizon():
    """Analysis-mode datasets must not expose samples whose forcing windows
    overrun the available forcing time axis."""

    class ShortForcingDatastore(DummyDatastore):
        def get_dataarray(self, category, split, **kwargs):
            da = super().get_dataarray(category=category, split=split, **kwargs)
            if category == "forcing":
                return da.isel(time=slice(None, -1))
            return da

    datastore = ShortForcingDatastore(n_grid_points=4, n_timesteps=7)
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=1,
        num_future_forcing_steps=2,
        standardize=False,
    )

    assert len(dataset) == 1

    _, _, forcing, _ = dataset[0]
    assert not torch.isnan(forcing).any()

    with pytest.raises(IndexError):
        dataset[1]


def test_weather_dataset_no_forcing_standardize():
    """Regression test: WeatherDataset must not raise AttributeError when the
    datastore has no forcing data and standardize=True (the default).

    Before the fix, self.da_forcing_std was accessed at line 123 of
    weather_dataset.py without ever being assigned when da_forcing is None,
    causing:
        AttributeError: 'WeatherDataset' object has no attribute
        'da_forcing_std'
    """

    class NoForcingDatastore(DummyDatastore):
        """DummyDatastore that returns None for the forcing category."""

        def get_dataarray(self, category, split, **kwargs):
            if category == "forcing":
                return None
            return super().get_dataarray(
                category=category, split=split, **kwargs
            )

    datastore = NoForcingDatastore(n_grid_points=100, n_timesteps=20)

    # Should not raise AttributeError
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=3,
        standardize=True,
    )

    assert dataset.forcing_std_safe is None
    assert dataset.da_forcing_mean is None
    assert dataset.da_forcing_std is None

    _, _, forcing, _ = dataset[0]
    assert forcing.shape[-1] == 0
