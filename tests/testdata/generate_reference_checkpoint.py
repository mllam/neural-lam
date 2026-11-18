"""Reproduce minimal reference checkpoint for backward compatibility tests.

Reference Checkpoint Metadata:
------------------------------
Filename: danra_winds_graph_lam_1level_v1.ckpt
Git Tag: v0.7.0
File Size: ~103 KB
SHA-256: 75ba32cb7b4d1a13dc36033a20ccb3edfae2c3d9f520d637b0d89be3b48c9555
S3 Path:
mllam-testdata/neural-lam/model-checkpoints/v0.7.0/danra_winds_graph_lam_1level_v1.ckpt

This script records the command sequence used to generate the minimal
reference model from a clean checkout of release tag ``v0.7.0``. It can also be
executed directly to regenerate the checkpoint locally.
"""

# Standard library
import os
import subprocess
import sys

# First-party
import neural_lam

REFERENCE_CHECKPOINT_KEY = (
    "mllam-testdata/neural-lam/model-checkpoints/v0.7.0/"
    "danra_winds_graph_lam_1level_v1.ckpt"
)
REFERENCE_CHECKPOINT_HASH = (
    "75ba32cb7b4d1a13dc36033a20ccb3edfae2c3d9f520d637b0d89be3b48c9555"
)


def generate_reference_checkpoint(output_dir: str = "./runs") -> None:
    """Run graph creation and training commands for reference checkpoint.

    Parameters
    ----------
    output_dir : str, default "./runs"
        Root directory where training runs and checkpoints will be saved.
    """
    version = getattr(neural_lam, "__version__", "unknown")
    if not str(version).startswith("0.7.0"):
        print(
            f"WARNING: Installed neural_lam version is '{version}'. "
            "Reference checkpoints should be generated from release tag "
            "'v0.7.0' to ensure authentic backward-compatibility testing."
        )

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    # Step 1: Create the 1level graph for DANRA test fixture
    create_graph_cmd = [
        sys.executable,
        "-m",
        "neural_lam.create_graph",
        "--config_path",
        "tests/datastore_examples/mdp/danra_100m_winds/config.yaml",
        "--name",
        "1level",
        "--levels",
        "1",
    ]
    print(f"Running: {' '.join(create_graph_cmd)}")
    subprocess.run(create_graph_cmd, check=True, env=env)

    # Step 2: Train 1-epoch minimal model
    train_cmd = [
        sys.executable,
        "-m",
        "neural_lam.train_model",
        "--config_path",
        "tests/datastore_examples/mdp/danra_100m_winds/config.yaml",
        "--model",
        "graph_lam",
        "--graph",
        "1level",
        "--hidden_dim",
        "4",
        "--hidden_layers",
        "1",
        "--processor_layers",
        "1",
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--ar_steps_train",
        "1",
        "--ar_steps_eval",
        "1",
        "--val_steps_to_log",
        "1",
        "--val_interval",
        "1",
        "--num_sanity_val_steps",
        "0",
        "--n_example_pred",
        "0",
        "--runs_root",
        output_dir,
    ]
    print(f"Running: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True, env=env)
    print(
        f"Checkpoint generated under {output_dir}/"
        "train-graph_lam-*/checkpoints/min_val_loss.ckpt"
    )


if __name__ == "__main__":
    generate_reference_checkpoint()
