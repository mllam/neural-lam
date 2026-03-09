# Third-party
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.graph_lam import GraphLAM
from tests.conftest import init_datastore_example


def test_graph_lam_inference_deterministic():

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    datastore = init_datastore_example("mdp")

    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        graph = "1level"
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 2
        mesh_aggr = "sum"
        lr = 1e-3
        val_steps_to_log = [1]
        metrics_watch = []
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1

    args = ModelArgs()

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        )
    )

    model = GraphLAM(
        args=args,
        datastore=datastore,
        config=config,
    )

    model.eval()

    mesh_nodes = model.mesh_static_features.shape[0]
    x = torch.randn(1, mesh_nodes, args.hidden_dim)

    with torch.no_grad():
        out1 = model.process_step(x)
        out2 = model.process_step(x)

    torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-6)
