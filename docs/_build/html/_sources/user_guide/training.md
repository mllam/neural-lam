# Training Models

Models are trained using the {func}`~neural_lam.train_model` module:

```bash
python -m neural_lam.train_model --config_path <config-path>
```

Run `python -m neural_lam.train_model --help` for a full list of options.

## Key Training Options

| Option | Description |
|--------|-------------|
| `--config_path` | Path to the Neural-LAM configuration file |
| `--model` | Which model to train (`graph_lam`, `hi_lam`, `hi_lam_parallel`) |
| `--graph` | Which graph to use (e.g. `1level`, `multiscale`, `hierarchical`) |
| `--epochs` | Number of training epochs |
| `--processor_layers` | Number of GNN layers in the processor |
| `--ar_steps_train` | Autoregressive steps during training |
| `--ar_steps_eval` | Autoregressive steps during validation |

Checkpoints are saved in the `saved_models` directory.

## Available Models

### Graph-LAM

The basic graph-based LAM model using encode-process-decode with a mesh graph.
Used for both **L1-LAM** and **GC-LAM** (differing only by graph type).

```bash
# Train 1L-LAM
python -m neural_lam.train_model --model graph_lam --graph 1level ...

# Train GC-LAM
python -m neural_lam.train_model --model graph_lam --graph multiscale ...
```

### Hi-LAM

Hierarchical version of Graph-LAM with **sequential** message passing through the
mesh hierarchy during processing.

```bash
python -m neural_lam.train_model --model hi_lam --graph hierarchical ...
```

### Hi-LAM-Parallel

Hierarchical model where all message passing (up, down, inter-level) runs **in parallel**.

```bash
python -m neural_lam.train_model --model hi_lam_parallel --graph hierarchical ...
```

## High Performance Computing

Neural-LAM uses PyTorch Lightning's `DDP` backend for distributed training across
multiple GPU nodes.

### SLURM Example

```bash
#!/bin/bash -l
#SBATCH --job-name=Neural-LAM
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres:gpu=4
#SBATCH --partition=normal
#SBATCH --mem=444G
#SBATCH --no-requeue
#SBATCH --exclusive

conda activate neural-lam

srun -ul python -m neural_lam.train_model \
    --config_path /path/to/config.yaml \
    --num_nodes $SLURM_JOB_NUM_NODES
```

### Without SLURM

Select specific GPUs with:
```bash
python -m neural_lam.train_model --devices 0 1 ...
```
