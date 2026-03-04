# Standard library
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# Third-party
import torch

# First-party
from neural_lam.config import load_config_and_datastore
from neural_lam.train_model import MODELS


def build_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # FULL minimal training-compatible args
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="graph_lam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load", type=str)
    parser.add_argument("--restore_opt", action="store_true")
    parser.add_argument("--precision", type=str, default=32)
    parser.add_argument("--graph", type=str, default="1level")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--hidden_layers", type=int, default=1)
    parser.add_argument("--processor_layers", type=int, default=4)
    parser.add_argument("--mesh_aggr", type=str, default="sum")
    parser.add_argument("--output_std", action="store_true")
    parser.add_argument("--ar_steps_train", type=int, default=1)
    parser.add_argument("--loss", type=str, default="wmse")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--eval", type=str)
    parser.add_argument("--ar_steps_eval", type=int, default=10)
    parser.add_argument("--n_example_pred", type=int, default=1)
    parser.add_argument("--logger", type=str, default="wandb")
    parser.add_argument("--logger-project", type=str, default="neural_lam")
    parser.add_argument("--logger_run_name", type=str)
    parser.add_argument("--val_steps_to_log", nargs="+", type=int, default=[1])
    parser.add_argument("--metrics_watch", nargs="+", default=[])
    parser.add_argument("--var_leads_metrics_watch", type=str, default="{}")
    parser.add_argument("--num_past_forcing_steps", type=int, default=1)
    parser.add_argument("--num_future_forcing_steps", type=int, default=1)

    args = parser.parse_args(
        [
            "--config_path",
            "tests/datastore_examples/mdp/danra_100m_winds/config.yaml",
            "--graph",
            "1level",
        ]
    )

    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }

    return args


def test_graph_lam_regression():

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    # Load frozen artifacts
    # Standard library
    from pathlib import Path

    BASE_DIR = Path(__file__).parent / "regression"

    batch_input = torch.load(BASE_DIR / "reference_input.pt")
    reference_output = torch.load(BASE_DIR / "reference_output.pt")

    # Rebuild model exactly like freeze script
    args = build_args()
    config, datastore = load_config_and_datastore(args.config_path)

    ModelClass = MODELS[args.model]
    model = ModelClass(args, config=config, datastore=datastore)
    model.eval()

    with torch.no_grad():
        new_output = model.process_step(batch_input)

    torch.testing.assert_close(
        new_output,
        reference_output,
        rtol=1e-4,
        atol=1e-5,
    )
