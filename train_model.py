# Standard library
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed

# First-party
from neural_lam import constants, utils
from neural_lam.era5_dataset import ERA5Dataset
from neural_lam.forecast_to_xarr import forecast_to_xarr
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast
from neural_lam.weather_dataset import WeatherDataset

MODELS = {
    "graphcast": GraphCast,
    "graph_fm": GraphFM,
    "graph_efm": GraphEFM,
}


def main():
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )

    # General options
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_example",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=16,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Numerical precision to use for model (32/16/bf16-mixed) "
        "(default: bf16-mixed)",
    )
    parser.add_argument(
        "--sanity_batches",
        type=int,
        default=2,
        help="Number of validation batches to run in sanity checking step "
        "set to 0 to disable sanity checking (default: 2)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Dimensionality of latent R.V. at each node (if different than"
        " hidden_dim) (default: None (same as hidden_dim))",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (for prob. model: in "
        "decoder) (default: 4)",
    )
    parser.add_argument(
        "--encoder_processor_layers",
        type=int,
        default=2,
        help="Number of on-mesh GNN layers in encoder GNN (default: 2)",
    )
    parser.add_argument(
        "--prior_processor_layers",
        type=int,
        default=2,
        help="Number of on-mesh GNN layers in prior GNN (default: 2)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        type=int,
        default=0,
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: 0 (no))",
    )
    parser.add_argument(
        "--prior_dist",
        type=str,
        default="isotropic",
        help="Structure of Gaussian distribution in prior network output "
        "(isotropic/diagonal) (default: isotropic)",
    )
    parser.add_argument(
        "--learn_prior",
        type=int,
        default=1,
        help="If the prior should be learned as a mapping from previous state "
        "and forcing, otherwise static with mean 0 (default: 1 (yes))",
    )
    parser.add_argument(
        "--vertical_propnets",
        type=int,
        default=0,
        help="If PropagationNets should be used for all vertical message "
        "passing (g2m, m2g, up in hierarchy), in deterministic models."
        "(default: 0 (no))",
    )

    # Training options
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss (1-19) "
        "(default: 1)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=6,
        help="Step length in hours to consider single time step 1-3 "
        "(default: 6)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=1.0,
        help="Beta weighting in front of kl-term in ELBO (default: 1)",
    )
    parser.add_argument(
        "--crps_weight",
        type=float,
        default=0,
        help="Weighting for CRPS term of loss, not computed if = 0. CRPS is "
        "computed based on trajectories sampled using prior distribution. "
        "(default: 0)",
    )
    parser.add_argument(
        "--sample_obs_noise",
        type=int,
        default=0,
        help="If observation noise should be sampled during rollouts (both "
        "training and eval), or just mean prediction used "
        "(default: 0 (no))",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--save_forecasts",
        type=int,
        help="Save forecasts to Xarray for later evaluation, rather than "
        "directly computing evaluation metrics (default: 0 (not))",
    )
    parser.add_argument(
        "--save_vars",
        type=str,
        help="Which variables to save when save_forecasts=True. Should be "
        "comma-separated list of short variable names. If None, save all."
        "(default: None)",
    )
    parser.add_argument(
        "--save_levels",
        type=str,
        help="Which pressure levels to save when save_forecasts=True. Should be"
        " comma-separated list of integer pressure level. If None, save all. "
        "Ground level variables are always saved if not filtered out."
        "(default: None)",
    )
    parser.add_argument(
        "--expanded_test",
        type=int,
        default=0,
        help="Eval model on larger test set (2020-2023), rather than only 2020"
        "(default: 0 (only 2020))",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during val/test "
        "(default: 1)",
    )
    parser.add_argument(
        "--eval_leads",
        type=int,
        default=40,
        help="Number of time steps to predict and evaluat at during val/test"
        "(default: 40)",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of ensemble members during evaluation (default: 5)",
    )
    args = parser.parse_args()

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load data
    if args.dataset.startswith("global"):
        ds_class = ERA5Dataset
    else:  # LAM
        assert args.step_length <= 3, "Too high step length"
        ds_class = WeatherDataset

    train_loader = torch.utils.data.DataLoader(
        ds_class(
            args.dataset,
            pred_length=args.ar_steps,
            split="train",
            subsample_step=args.step_length,
        ),
        args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        ds_class(
            args.dataset,
            pred_length=args.eval_leads,
            split="val",
            subsample_step=args.step_length,
        ),
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    if args.load:
        model = model_class.load_from_checkpoint(args.load, args=args)
        if args.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        model = model_class(args)

    prefix = ""
    if args.eval:
        prefix = f"eval-{args.eval}-"
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    # Callbacks for saving model checkpoint
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_mean_loss",
            mode="min",
            save_last=True,
        )
    )
    # Save checkpoints for minimum loss at specific lead times
    # Only include lead times actually forecasted
    checkpoint_times = constants.VAL_STEP_CHECKPOINTS[
        constants.VAL_STEP_CHECKPOINTS <= args.eval_leads
    ]
    for unroll_time in checkpoint_times:
        metric_name = f"val_loss_unroll{unroll_time}"
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=f"saved_models/{run_name}",
                filename=f"min_{metric_name}",
                monitor=metric_name,
                mode="min",
            )
        )
    logger = pl.loggers.WandbLogger(
        project=constants.WANDB_PROJECT, name=run_name, config=args
    )

    # Training strategy
    # If doing pure autoencoder training (kl_beta = 0), the prior network is not
    # used at all in producing the loss. This is desired, but DDP complains.
    strategy = "ddp" if args.kl_beta > 0 else "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy=strategy,
        accelerator=device_name,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        num_sanity_val_steps=args.sanity_batches,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(logger)  # Do after wandb.init

    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            eval_loader = torch.utils.data.DataLoader(
                ds_class(
                    args.dataset,
                    pred_length=args.eval_leads,
                    split="test",
                    subsample_step=args.step_length,
                    expanded_test=bool(args.expanded_test),
                ),
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
            )

        print(f"Running evaluation on {args.eval}")
        if args.save_forecasts:
            print("Saving eval forecasts to zarr")
            assert args.load, "Need to load a model to save forecasts from"
            load_name_cleaned = args.load.replace("/", "_")  # Replace /
            fc_save_name = f"{args.dataset}-{args.eval}-{load_name_cleaned}"
            forecast_to_xarr(
                model,
                eval_loader,
                fc_save_name,
                device_name,
                var_filter=args.save_vars,
                level_filter=args.save_levels,
                ens_size=args.ensemble_size,
            )
            print("Forecasts saved")
        else:
            trainer.test(model=model, dataloaders=eval_loader)
    else:
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )


if __name__ == "__main__":
    main()
