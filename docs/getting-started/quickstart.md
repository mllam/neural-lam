# Quickstart

This runs the minimal Neural-LAM workflow from the command line on the small DANRA example dataset in `tests/datastore_examples/mdp/danra_100m_winds`.
The {doc}`DANRA tutorial <../notebooks/hello_world_danra>` runs the same steps in Python and shows the resulting graph, metrics and forecasts.

Run the commands from the repository root after the {doc}`installation`.
The datastore downloads and prepares its data on first use, and W&B is switched to offline mode so nothing is uploaded (see {doc}`../user-guide/logging`):

```bash
export WANDB_MODE=offline
CONFIG=tests/datastore_examples/mdp/danra_100m_winds/config.yaml
```

## 1. Create a graph

```bash
python -m neural_lam.create_graph_with_wmg --config_path $CONFIG --archetype keisler --name keisler
```

See {doc}`../user-guide/graphs` for the other archetypes and their options.

## 2. Train a model

```bash
python -m neural_lam.train_model --config_path $CONFIG --model graph_lam --graph keisler --epochs 1 --logger_run_name quickstart
```

See {doc}`../user-guide/training` for the available models and training options, and {doc}`../user-guide/datastores` for how to write a config for your own data.

## 3. Evaluate the model

```bash
python -m neural_lam.train_model --eval test --config_path $CONFIG --model graph_lam --graph keisler --load runs/quickstart/checkpoints/min_val_loss.ckpt
```

See {doc}`../user-guide/training` for the evaluation options.
