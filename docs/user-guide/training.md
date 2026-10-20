# Training and Evaluation

## Training a model

```bash
python -m neural_lam.train_model \
    --config_path <config_path> \
    --model <model> \
    --graph <graph-name>
```

Run `python -m neural_lam.train_model --help` for the full list of options.
Key flags:

| Flag | Description |
|---|---|
| `--config_path` | Path to the neural-lam config file |
| `--model` | Model class: `graph_lam`, `hi_lam`, `hi_lam_parallel`, `persistence` |
| `--graph` | Name of the graph to use (must match a directory under `graphs/`) |
| `--epochs` | Number of training epochs |
| `--processor_layers` | Number of GNN layers in the processor |
| `--ar_steps_train` | Autoregressive steps for training loss |
| `--ar_steps_eval` | Autoregressive steps during validation |

Checkpoints are saved under `runs/<run-name>/checkpoints/`:

- `min_val_loss.ckpt` — best validation loss checkpoint
- `last.ckpt` — end-of-epoch rescue checkpoint (useful for HPC jobs)

## Available models

### Graph-LAM (`graph_lam`)

Basic encode-process-decode model with a flat mesh graph. Used for both
L1-LAM (single-level graph) and GC-LAM (multiscale graph).

```bash
python -m neural_lam.train_model \
    --config_path <config> --model graph_lam --graph keisler
```

### Hi-LAM (`hi_lam`)

Hierarchical mesh graph with sequential message passing up and down through
the hierarchy during processing.

```bash
python -m neural_lam.train_model \
    --config_path <config> --model hi_lam --graph hierarchical
```

### Hi-LAM-Parallel (`hi_lam_parallel`)

Variant of Hi-LAM where all message passing in the hierarchy (up, down,
inter-level) runs in parallel.

```bash
python -m neural_lam.train_model \
    --config_path <config> --model hi_lam_parallel --graph hierarchical
```

### Persistence (`persistence`)

Trivial baseline that returns the previous state unchanged at each step.
Has no learnable parameters so it must be evaluated directly without training:

```bash
python -m neural_lam.train_model \
    --eval val --config_path <config> --model persistence
```

## Evaluating a model

Evaluation reuses the same `train_model` entry point with the `--eval` flag:

```bash
python -m neural_lam.train_model \
    --eval test \
    --config_path <config> \
    --model graph_lam \
    --graph keisler \
    --load runs/<run-name>/checkpoints/min_val_loss.ckpt
```

Use `--eval val` to evaluate on the validation set, `--eval test` for the test
set. Key evaluation flags:

| Flag | Description |
|---|---|
| `--load` | Path to a checkpoint file (`.ckpt`) |
| `--n_example_pred` | Number of example predictions to plot |
| `--ar_steps_eval` | Autoregressive steps during evaluation |

> **Note:** Avoid using multiple GPUs for evaluation. The `DistributedSampler`
> replicates samples to keep batch sizes equal across devices, making
> evaluation metrics unreliable. Use a single GPU or batch size 1.

## Multi-GPU and HPC training

Neural-LAM uses PyTorch Lightning's DDP backend for distributed training.

```bash
python -m neural_lam.train_model \
    --config_path <config> \
    --num_nodes $SLURM_JOB_NUM_NODES
```

Example SLURM job script:

```bash
#!/bin/bash -l
#SBATCH --job-name=Neural-LAM
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=normal
#SBATCH --mem=444G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=runs/slurm_logs/neurallam_out_%j.log
#SBATCH --error=runs/slurm_logs/neurallam_err_%j.log

conda activate neural-lam

srun -ul python -m neural_lam.train_model \
    --config_path /path/to/config.yaml \
    --num_nodes $SLURM_JOB_NUM_NODES
```

On a system without SLURM, select specific GPUs with `--devices`, e.g.
`--devices 0 1` to use the first two GPUs.

> **DGX Spark / container compatibility:** see
> [issue #163](https://github.com/mllam/neural-lam/issues/163) for known-working
> and known-failing PyTorch / CUDA / container combinations.

## Python API

Training and evaluation are also accessible programmatically via
{py:mod}`neural_lam.api`:

```python
from neural_lam import api

run = api.train(config_path="data/config.yaml", model="graph_lam", graph="keisler")
print(run.checkpoint_path)

api.evaluate(config_path="data/config.yaml", model="graph_lam",
             graph="keisler", load=run.checkpoint_path)
```
