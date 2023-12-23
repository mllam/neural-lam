import os
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from pytorch_lightning.utilities import rank_zero_only

import wandb
from neural_lam import constants, utils
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.hi_lam_parallel import HiLAMParallel
from neural_lam.weather_dataset import WeatherDataModule

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


@rank_zero_only
def print_args(args):
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


@rank_zero_only
def print_eval(args_eval):
    print(f"Running evaluation on {args_eval}")


@rank_zero_only
def init_wandb(args):
    if args.resume_run is None:
        prefix = "subset-" if args.subset_ds else ""
        if args.eval:
            prefix = prefix + f"eval-{args.eval}-"
        run_name = f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"\
            f"{time.strftime('%m_%d_%H_%M_%S')}"
        wandb.init(
            name=run_name,
            project=constants.wandb_project,
            config=args,
        )
        logger = pl.loggers.WandbLogger(project=constants.wandb_project, name=run_name,
                                        config=args)
    else:
        wandb.init(
            project=constants.wandb_project,
            config=args,
            id=args.resume_run,
            resume='must'
        )
        logger = pl.loggers.WandbLogger(
            project=constants.wandb_project,
            id=args.resume_run,
            config=args)

    return logger


def main():
    # if torch.cuda.is_available():
    #     init_process_group(backend="nccl")
    parser = ArgumentParser(description='Train or evaluate NeurWP models for LAM')

    # General options
    parser.add_argument(
        '--dataset', type=str, default="meps_example",
        help='Dataset, corresponding to name in data directory (default: meps_example)')
    parser.add_argument(
        '--model', type=str, default="graph_lam",
        help='Model architecture to train/evaluate (default: graph_lam)')
    parser.add_argument(
        '--subset_ds', type=int, default=0,
        help='Use only a small subset of the dataset, for debugging (default: 0=false)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of workers in data loader (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit (default: 200)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument('--load', type=str,
                        help='Path to load model parameters from (default: None)')
    parser.add_argument('--resume_run', type=str,
                        help='Run ID to resume (default: None)')
    parser.add_argument(
        '--precision', type=str, default=32,
        help='Numerical precision to use for model (32/16/bf16) (default: 32)')

    # Model architecture
    parser.add_argument(
        '--graph', type=str, default="multiscale",
        help='Graph to load and use in graph-based model (default: multiscale)')
    parser.add_argument(
        '--hidden_dim', type=int, default=64,
        help='Dimensionality of all hidden representations (default: 64)')
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='Number of hidden layers in all MLPs (default: 1)')
    parser.add_argument('--processor_layers', type=int, default=4,
                        help='Number of GNN layers in processor GNN (default: 4)')
    parser.add_argument(
        '--mesh_aggr', type=str, default="sum",
        help='Aggregation to use for m2m processor GNN layers (sum/mean) (default: sum)')

    # Training options
    parser.add_argument(
        '--ar_steps', type=int, default=1,
        help='Number of steps to unroll prediction for in loss (1-24) (default: 1)')
    parser.add_argument('--loss', type=str, default="mse",
                        help='Loss function to use (default: mse)')
    parser.add_argument(
        '--step_length', type=int, default=1,
        help='Step length in hours to consider single time step 1-3 (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument(
        '--val_interval', type=int, default=1,
        help='Number of epochs training between each validation run (default: 1)')

    # Evaluation options
    parser.add_argument(
        '--eval', type=str, default=None,
        help='Eval model on given data split (val/test) (default: None (train model))')
    parser.add_argument(
        '--n_example_pred', type=int, default=1,
        help='Number of example predictions to plot during evaluation (default: 1)')
    args = parser.parse_args()
    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (None, "val", "test"), f"Unknown eval setting: {args.eval}"
    assert args.loss in ("mse", "mae", "huber"), f"Unknown loss function: {args.loss}"

    # Set seed
    seed.seed_everything(args.seed)

    # Create datamodule
    data_module = WeatherDataModule(
        args.dataset,
        subset=bool(args.subset_ds),
        batch_size=args.batch_size,
        num_workers=args.n_workers
    )

    # Get the device for the current process
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")  # Allows using Tensor Cores on A100s

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    model = model_class(args)

    result = init_wandb(args)
    if result is not None:
        logger = result
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=logger.experiment.dir,
            filename="latest",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        )
    else:
        logger = None
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath="saved_models",
            filename="latest",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        )

    if args.eval:
        use_distributed_sampler = False
    else:
        use_distributed_sampler = True

    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = torch.cuda.device_count()
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
    else:
        accelerator = "cpu"
        devices = 1
        num_nodes = 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback] if checkpoint_callback is not None else [],
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        use_distributed_sampler=use_distributed_sampler,
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        profiler="simple",
        # num_sanity_val_steps=0
        # strategy="ddp",
        # deterministic=True,
        # limit_val_batches=0
        # fast_dev_run=True
    )
    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(logger)  # Do after wandb.init
    if args.eval:
        print_eval(args.eval)
        if args.eval == "val":
            data_module.split = "val"
        else:  # Test
            data_module.split = "test"
        trainer.test(model=model, datamodule=data_module, ckpt_path=args.load)
    else:
        # Train model
        data_module.split = "train"
        if args.load:
            trainer.fit(model=model, datamodule=data_module, ckpt_path=args.load)
        else:
            trainer.fit(model=model, datamodule=data_module)

    # Print profiler
    print(trainer.profiler)


if __name__ == "__main__":
    main()
