# Training and Evaluation

## Train Models

Models can be trained using `python -m neural_lam.train_model --config_path <config_path>`.
Run `python -m neural_lam.train_model --help` for a full list of training options.
A few of the key ones are outlined below:

* `--config_path`: Path to the configuration for neural-lam (for example in `data/myexperiment/config.yaml`).
* `--model`: Which model to train
* `--graph`: Which graph to use with the model
* `--epochs`: Number of epochs to train for
* `--processor_layers`: Number of GNN layers to use in the processing part of the model
* `--ar_steps_train`: Number of time steps to unroll for when making predictions and computing the loss
* `--ar_steps_eval`: Number of time steps to unroll for during validation steps

Checkpoints of trained models are stored under `runs/<run-name>/checkpoints/`, alongside Lightning and logger outputs for the same run.
The implemented models are:

### Graph-LAM

This is the basic graph-based LAM model.
The encode-process-decode framework is used with a mesh graph in order to make one-step predictions.
This model class is used both for the L1-LAM and GC-LAM models from the [paper](https://arxiv.org/abs/2309.17370), only with different graphs.

To train 1L-LAM use
```
python -m neural_lam.train_model --model graph_lam --graph 1level ...
```

To train GC-LAM use
```
python -m neural_lam.train_model --model graph_lam --graph multiscale ...
```

### Hi-LAM

A version of Graph-LAM that uses a hierarchical mesh graph and performs sequential message passing through the hierarchy during processing.

To train Hi-LAM use
```
python -m neural_lam.train_model --model hi_lam --graph hierarchical ...
```

### Hi-LAM-Parallel

A version of Hi-LAM where all message passing in the hierarchical mesh (up, down, inter-level) is ran in parallel.
Not included in the paper as initial experiments showed worse results than Hi-LAM, but could be interesting to try in more settings.

To train Hi-LAM-Parallel use
```
python -m neural_lam.train_model --model hi_lam_parallel --graph hierarchical ...
```

### Persistence

A trivial baseline model that returns the previous state unchanged at each step (`--model persistence`). Since it has no learnable parameters, it must be evaluated directly without training (e.g. `--eval val` or `--eval test`).

Checkpoint files for our models trained on the MEPS data are available upon request.

### High Performance Computing

The training script can be run on a cluster with multiple GPU-nodes. Neural LAM is set up to use PyTorch Lightning's `DDP` backend for distributed training.
The code can be used on systems both with and without slurm. If the cluster has multiple nodes, set the `--num_nodes` argument accordingly.

Using SLURM, the job can be started with `sbatch slurm_job.sh` with a shell script like the following.
```
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

# Load necessary modules or activate environment, for example:
conda activate neural-lam

srun -ul python -m neural_lam.train_model \
    --config_path /path/to/config.yaml \
    --num_nodes $SLURM_JOB_NUM_NODES
```

When using on a system without SLURM, where all GPU's are visible, it is possible to select a subset of GPU's to use for training with the `devices` cli argument, e.g. `--devices 0 1` to use the first 2 GPU's.

> **DGX Spark / container compatibility:** see [issue #163](https://github.com/mllam/neural-lam/issues/163) for a list of currently known-working and known-failing PyTorch / CUDA / container combinations on DGX Spark.

## Evaluate Models

Evaluation is also done using `python -m neural_lam.train_model --config_path <config-path>`, but using the `--eval` option.
Use `--eval val` to evaluate the model on the validation set and `--eval test` to evaluate on test data.
Most of the training options are also relevant for evaluation.
Some options specifically important for evaluation are:

* `--load`: Path to model checkpoint file (`.ckpt`) to load parameters from
* `--n_example_pred`: Number of example predictions to plot during evaluation.
* `--ar_steps_eval`: Number of time steps to unroll for during evaluation

For example, for a `graph_lam` model trained on the `keisler` graph (`--model` and `--graph` must match the ones used for training):

```bash
python -m neural_lam.train_model \
    --eval test \
    --config_path <config> \
    --model graph_lam \
    --graph keisler \
    --load <checkpoint-path>
```

**Note:** While it is technically possible to use multiple GPUs for running evaluation, this is strongly discouraged. If using multiple devices the `DistributedSampler` will replicate some samples to make sure all devices have the same batch size, meaning that evaluation metrics will be unreliable.
A possible workaround is to just use batch size 1 during evaluation.
This issue stems from PyTorch Lightning. See for example [this PR](https://github.com/Lightning-AI/torchmetrics/pull/1886) for more discussion.
