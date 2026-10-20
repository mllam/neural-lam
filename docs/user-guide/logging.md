# Experiment Logging

Neural-LAM supports two logging backends: Weights & Biases (default) and
MLflow. Select the backend with `--logger <backend>` when calling
`train_model`.

## Weights & Biases

[Weights & Biases](https://www.wandb.ai/) is the default logger. When enabled,
training configuration, training/test statistics and plots are sent to the W&B
servers and available in an interactive web interface. When disabled, everything
is saved locally under `runs/<run-name>/wandb/`.

Log in before training:

```bash
wandb login
```

To disable W&B and log locally instead:

```bash
wandb off
```

The W&B project name defaults to `neural-lam` and can be changed with the
`--wandb_project` flag. To resume an existing run (e.g. after an HPC
interruption), pass its ID with `--wandb_id <run-id>`.

See the [W&B documentation](https://docs.wandb.ai/) for further details.

## MLflow

[MLflow](https://mlflow.org/) is an alternative self-hosted logger. Enable it
with:

```bash
python -m neural_lam.train_model --config_path <config> --logger mlflow
```

Set the tracking server URI via the `MLFLOW_TRACKING_URI` environment variable:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 \
    python -m neural_lam.train_model --config_path <config> --logger mlflow
```

See the [MLflow documentation](https://mlflow.org/docs/latest/index.html) for
how to run a tracking server locally or on a remote host.
