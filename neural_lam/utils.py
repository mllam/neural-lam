# Standard library
import os
import shutil
import subprocess
import tempfile
import warnings
from functools import cache
from pathlib import Path

# Third-party
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from tueplots import bundles, figsizes

# Local
from .custom_loggers import CustomMLFlowLogger


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def make_mlp(blueprint, layer_norm=True):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    return nn.Sequential(*layers)


@cache
def has_working_latex():
    """
    Check if LaTeX is available or its toolchain
    """
    # If latex/toolchain is not available, some visualizations might not render
    # correctly, but will at least not raise an error. Alternatively, use
    # unicode raised numbers.

    if not shutil.which("latex"):
        return False
    if not shutil.which("dvipng"):
        return False
    if not (
        shutil.which("gs")
        or shutil.which("gswin64c")
        or shutil.which("gswin32c")
    ):
        return False

    tex_src = r"""
\documentclass{article}
\usepackage{fix-cm}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\begin{document}
$E=mc^2$ \LaTeX\ ok
\end{document}
""".lstrip()

    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "test.tex").write_text(tex_src, encoding="utf-8")
            cmd = [
                "latex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "test.tex",
            ]
            subprocess.run(
                cmd,
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=True,
            )
            cmd_dvipng = ["dvipng", "-D", "100", "-o", "test.png", "test.dvi"]
            subprocess.run(
                cmd_dvipng,
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=True,
            )
        return True
    except Exception:
        return False


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """

    usetex = has_working_latex()
    bundle = bundles.neurips2023(usetex=usetex, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    """Print only from rank 0 process"""
    print(*args, **kwargs)


def init_training_logger_metrics(training_logger, val_steps):
    """
    Set up logger metrics to track
    """
    experiment = training_logger.experiment
    if isinstance(training_logger, WandbLogger):
        experiment.define_metric("val_mean_loss", summary="min")
        for step in val_steps:
            experiment.define_metric(f"val_loss_unroll{step}", summary="min")
    elif isinstance(training_logger, MLFlowLogger):
        pass
    else:
        warnings.warn(
            "Only WandbLogger & MLFlowLogger is supported for tracking metrics.\
             Experiment results will only go to stdout."
        )


@rank_zero_only
def setup_training_logger(datastore, args, run_name):
    """

    Parameters
    ----------
    datastore : Datastore
        Datastore object.

    args : argparse.Namespace
        Arguments from command line.

    run_name : str
        Name of the run.

    Returns
    -------
    logger : pytorch_lightning.loggers.base
        Logger object.
    """

    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(
            project=args.logger_project,
            name=run_name,
            config=dict(training=vars(args), datastore=datastore._config),
        )
    elif args.logger == "mlflow":
        url = os.getenv("MLFLOW_TRACKING_URI")
        if url is None:
            raise ValueError(
                "MLFlow logger requires setting MLFLOW_TRACKING_URI in env."
            )
        logger = CustomMLFlowLogger(
            experiment_name=args.logger_project,
            tracking_uri=url,
            run_name=run_name,
        )
        logger.log_hyperparams(
            dict(training=vars(args), datastore=datastore._config)
        )

    return logger


def inverse_softplus(x, beta=1, threshold=20):
    """
    Inverse of torch.nn.functional.softplus

    Input is clamped to approximately positive values of x, and the function is
    linear for inputs above x*beta for numerical stability.

    Note that this torch.clamp will make gradients 0, but this is not a
    problem as values of x that are this close to 0 have gradients of 0 anyhow.
    """
    x_clamped = torch.clamp(
        x, min=torch.log(torch.tensor(1e-6 + 1)) / beta, max=threshold / beta
    )

    non_linear_part = torch.log(torch.expm1(x_clamped * beta)) / beta

    below_threshold = x * beta <= threshold

    x = torch.where(condition=below_threshold, input=non_linear_part, other=x)

    return x


def inverse_sigmoid(x):
    """
    Inverse of torch.sigmoid

    Sigmoid output takes values in [0,1], this makes sure input is just within
    this interval.
    Note that this torch.clamp will make gradients 0, but this is not a problem
    as values of x that are this close to 0 or 1 have gradients of 0 anyhow.
    """
    x_clamped = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    return torch.log(x_clamped / (1 - x_clamped))
