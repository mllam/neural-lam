# Logging your experiments

## Weights & Biases Integration

The project is fully integrated with [Weights & Biases](https://www.wandb.ai/) (W&B) for logging and visualization, but can just as easily be used without it.
When W&B is used, training configuration, training/test statistics and plots are sent to the W&B servers and made available in an interactive web interface.
If W&B is turned off, logging instead saves everything locally under the run directory (e.g. `runs/<run-name>/wandb/`).
The W&B project name is set to `neural_lam`, but this can be changed in the flags of `python -m neural_lam.train_model` (using argparse).
See the [W&B documentation](https://docs.wandb.ai/) for details.

If you would like to login and use W&B, run:
```
wandb login
```
If you would like to turn off W&B and just log things locally, run:
```
wandb off
```

To resume an existing W&B run (e.g. after an HPC interruption), pass its ID with `--wandb_id <run-id>`.

## MLFlow Integration

The project is also integrated with [MLFlow](https://mlflow.org/) for logging and storing artefacts.

MLFlow is not used by default, but can be switched to by setting `--logger mlflow` in the training command. With MLFlow enabled, training configuration, training/test statistics and plots are logged to the MLFlow server. MLFlow is self-hosted and can be run locally or on a server. See the [MLFlow documentation](https://mlflow.org/docs/latest/index.html) for details.

Use the environment variable `MLFLOW_TRACKING_URI` to set the URI of the MLFlow server. If not set the logging can not be used. An example of setting the URI to a server is and running a training command is `MLFLOW_TRACKING_URI=http://localhost:5000 python -m neural_lam.train_model --config_path <config_path> --logger mlflow`.
