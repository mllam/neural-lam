# Evaluation and Logging

## Evaluating Models

Evaluation uses the same entry point as training, with the `--eval` flag:

```bash
python -m neural_lam.train_model --config_path <config-path> --eval <split>
```

Use `--eval val` for validation set or `--eval test` for test data.

### Key Evaluation Options

| Option | Description |
|--------|-------------|
| `--load` | Path to model checkpoint (`.ckpt`) to load |
| `--n_example_pred` | Number of example predictions to plot |
| `--ar_steps_eval` | Number of autoregressive steps to unroll |

:::{warning}
Using multiple GPUs for evaluation is **strongly discouraged**. The
`DistributedSampler` replicates samples to equalize batch sizes across devices,
which makes evaluation metrics unreliable. Use batch size 1 if you must use
multiple devices.
:::

## Logging

### Weights & Biases

Neural-LAM is fully integrated with [Weights & Biases](https://www.wandb.ai/)
for experiment tracking. Training configuration, statistics, and plots are sent
to the W&B servers.

```bash
# Login to W&B
wandb login

# Or disable W&B (logs locally to wandb/dryrun...)
wandb off
```

The W&B project name defaults to `neural-lam` but can be changed via CLI flags.

### MLFlow

[MLFlow](https://mlflow.org/) integration is also available as an alternative logger:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 \
    python -m neural_lam.train_model \
    --config_path <config_path> \
    --logger mlflow
```

Set `MLFLOW_TRACKING_URI` to point to your MLFlow server. See the
[MLFlow documentation](https://mlflow.org/docs/latest/index.html) for setup details.

## Metrics

Neural-LAM implements several evaluation metrics in {mod}`neural_lam.metrics`:

| Metric | Function | Description |
|--------|----------|-------------|
| MSE | {func}`~neural_lam.metrics.mse` | Mean Squared Error |
| MAE | {func}`~neural_lam.metrics.mae` | Mean Absolute Error |
| WMSE | {func}`~neural_lam.metrics.wmse` | Weighted MSE |
| WMAE | {func}`~neural_lam.metrics.wmae` | Weighted MAE |
| NLL | {func}`~neural_lam.metrics.nll` | Negative Log Likelihood |
| CRPS | {func}`~neural_lam.metrics.crps_gauss` | Continuous Ranked Probability Score |
