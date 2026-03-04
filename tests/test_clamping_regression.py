import torch
from tests.dummy_datastore import DummyDatastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.config import NeuralLAMConfig
from neural_lam.create_graph import create_graph_from_datastore


def test_clamping_deterministic():
    torch.manual_seed(42)

    datastore = DummyDatastore()

    graph_name = "1level"
    graph_dir = datastore.root_path / "graph" / graph_name

    if not graph_dir.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir),
            n_max_levels=1,
        )

    class Args:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        graph = graph_name
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 1
        mesh_aggr = "sum"
        lr = 1e-3
        val_steps_to_log = []
        metrics_watch = []
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1

    config = NeuralLAMConfig(
        datastore={"kind": datastore.SHORT_NAME, "config_path": None}
    )

    model = GraphLAM(args=Args(), config=config, datastore=datastore)

    B = 1
    N = datastore.num_grid_points
    F = datastore.get_num_data_vars("state")

    prev_state = torch.randn(B, N, F)
    delta = torch.randn(B, N, F)

    clamped = model.get_clamped_new_state(delta, prev_state)

    assert torch.isfinite(clamped).all()