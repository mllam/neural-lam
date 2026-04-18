# Standard library
import functools
from pathlib import Path

# Third-party
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.models.graph_lam import GraphLAM
from tests.conftest import init_datastore_example


@functools.lru_cache(maxsize=1)
def _get_model():
    datastore = init_datastore_example(MDPDatastore.SHORT_NAME)

    graph_name = "1level"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

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
        lr = 1e-3
        val_steps_to_log = [1]
        metrics_watch = []
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1

    var1 = datastore.get_vars_names("state")[0]
    var2 = datastore.get_vars_names("state")[1]

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        ),
        training=nlconfig.TrainingConfig(
            output_clamping=nlconfig.OutputClamping(
                lower={var1: -1.0, var2: -1.0},
                upper={var1: 1.0, var2: 1.0},
            )
        ),
    )

    model = GraphLAM(ModelArgs(), config=config, datastore=datastore)
    return model, datastore


@given(st.data())
@settings(deadline=None, max_examples=10)
def test_clamping_property(data):
    model, datastore = _get_model()

    n_features = len(datastore.get_vars_names(category="state"))

    batch_size = data.draw(st.integers(min_value=1, max_value=8))
    n_grid = data.draw(st.integers(min_value=1, max_value=64))

    # Third-party
    import numpy as np
    from hypothesis.extra.numpy import arrays

    state_np = data.draw(
        arrays(
            dtype=np.float32,
            shape=(batch_size, n_grid, n_features),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                width=32,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    delta_np = data.draw(
        arrays(
            dtype=np.float32,
            shape=(batch_size, n_grid, n_features),
            elements=st.floats(
                min_value=-1e6,
                max_value=1e6,
                width=32,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    state = torch.from_numpy(state_np)
    delta = torch.from_numpy(delta_np)

    prediction = model.get_clamped_new_state(delta, state)

    if (
        hasattr(model, "clamp_lower_upper_idx")
        and model.clamp_lower_upper_idx.numel() > 0
    ):
        idx = model.clamp_lower_upper_idx
        lower = model.sigmoid_lower_lims.view(1, 1, -1)
        upper = model.sigmoid_upper_lims.view(1, 1, -1)
        assert (prediction[..., idx] >= lower).all().item(), "Below lower bound"
        assert (prediction[..., idx] <= upper).all().item(), "Above upper bound"

    if hasattr(model, "clamp_lower_idx") and model.clamp_lower_idx.numel() > 0:
        idx = model.clamp_lower_idx
        lower = model.softplus_lower_lims.view(1, 1, -1)
        assert (prediction[..., idx] >= lower).all().item(), "Below lower bound"

    if hasattr(model, "clamp_upper_idx") and model.clamp_upper_idx.numel() > 0:
        idx = model.clamp_upper_idx
        upper = model.softplus_upper_lims.view(1, 1, -1)
        assert (prediction[..., idx] <= upper).all().item(), "Above upper bound"
