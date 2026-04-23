"""Utility helpers shared across Neural-LAM training and evaluation."""

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
from loguru import logger
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
        """
        Register a collection of tensors as buffers inside a module.

        Parameters
        ----------
        buffer_tensors : Sequence[torch.Tensor]
            Buffers to register in the order they should be indexed.
        persistent : bool, optional
            If ``True``, buffers are saved in checkpoints. Default ``True``.
        """
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        """Return the buffer at ``key`` (0-indexed)."""
        return getattr(self, f"b{key}")

    def __len__(self):
        """Return the number of registered buffers."""
        return self.n_buffers

    def __iter__(self):
        """Iterate over the registered buffers in ascending index order."""
        return (self[i] for i in range(len(self)))

    def __itruediv__(self, other):
        """Divide each element in list with other"""
        return self.__imul__(1.0 / other)

    def __imul__(self, other):
        """Multiply each element in list with other"""
        for buffer_tensor in self:
            buffer_tensor *= other

        return self


def zero_index_edge_index(edge_index):
    """
    Make both sender and receiver indices of edge_index start at 0
    """
    return edge_index - edge_index.min(dim=1, keepdim=True)[0]


def zero_index_m2g(
    m2g_edge_index: torch.Tensor,
    mesh_static_features: list[torch.Tensor],
    mesh_first: bool,
    restore: bool = False,
) -> torch.Tensor:
    """
    Zero-index the m2g (mesh-to-grid) edge index, or undo this operation.

    Special handling is needed since not all mesh nodes may be present.

    Parameters
    ----------
    m2g_edge_index : torch.Tensor
        Edge index tensor of shape (2, N_edges).
    mesh_static_features : list of torch.Tensor
        Mesh node feature tensors.
    mesh_first : bool
        If True, mesh nodes are indexed before grid nodes.
    restore : bool
        If True, undo zero-indexing (restore original indices).

    Returns
    -------
    torch.Tensor
        Edge index tensor with zero-based or restored indices.
    """

    sign = 1 if restore else -1

    if mesh_first:
        # Mesh has the first indices, adjust grid indices (row 1)
        num_mesh_nodes = mesh_static_features[0].shape[0]
        return torch.stack(
            (
                m2g_edge_index[0],
                m2g_edge_index[1] + sign * num_mesh_nodes,
            ),
            dim=0,
        )
    else:
        # Grid (interior) has the first indices, adjust mesh indices (row 0)
        num_interior_nodes = m2g_edge_index[1].max() + 1
        return torch.stack(
            (
                m2g_edge_index[0] + sign * num_interior_nodes,
                m2g_edge_index[1],
            ),
            dim=0,
        )


def zero_index_g2m(
    g2m_edge_index: torch.Tensor,
    mesh_static_features: list[torch.Tensor],
    mesh_first: bool,
    restore: bool = False,
) -> torch.Tensor:
    """
    Zero-index the g2m (grid-to-mesh) edge index, or undo this operation.

    Special handling is needed since not all mesh nodes may be present.

    Parameters
    ----------
    g2m_edge_index : torch.Tensor
        Edge index tensor of shape (2, N_edges).
    mesh_static_features : list of torch.Tensor
        Mesh node feature tensors.
    mesh_first : bool
        If True, mesh nodes are indexed before grid nodes.
    restore : bool
        If True, undo zero-indexing (restore original indices).

    Returns
    -------
    torch.Tensor
        Edge index tensor with zero-based or restored indices.
    """

    sign = 1 if restore else -1

    if mesh_first:
        # Mesh has the first indices, adjust grid indices (row 0)
        num_mesh_nodes = mesh_static_features[0].shape[0]
        return torch.stack(
            (
                g2m_edge_index[0] + sign * num_mesh_nodes,
                g2m_edge_index[1],
            ),
            dim=0,
        )
    else:
        # Grid has the first indices, adjust mesh indices (row 1)
        num_grid_nodes = g2m_edge_index[0].max() + 1
        return torch.stack(
            (
                g2m_edge_index[0],
                g2m_edge_index[1] + sign * num_grid_nodes,
            ),
            dim=0,
        )


def load_graph(graph_dir_path, device="cpu"):
    """Load all tensors representing the graph from `graph_dir_path`.

    Needs the following files for all graphs:
    - m2m_edge_index.pt
    - g2m_edge_index.pt
    - m2g_edge_index.pt
    - m2m_features.pt
    - g2m_features.pt
    - m2g_features.pt
    - mesh_features.pt

    And in addition for hierarchical graphs:
    - mesh_up_edge_index.pt
    - mesh_down_edge_index.pt
    - mesh_up_features.pt
    - mesh_down_features.pt

    Parameters
    ----------
    graph_dir_path : str
        Path to directory containing the graph files.
    device : str
        Device to load tensors to.

    Returns
    -------
    hierarchical : bool
        Whether the graph is hierarchical.
    graph : dict
        Dictionary containing the graph tensors, with keys as follows:
        - g2m_edge_index
        - m2g_edge_index
        - m2m_edge_index
        - mesh_up_edge_index
        - mesh_down_edge_index
        - g2m_features
        - m2g_features
        - m2m_features
        - mesh_up_features
        - mesh_down_features
        - mesh_static_features

    """

    def loads_file(fn):
        """Load ``torch.load`` data from ``graph_dir_path``.

        Applies ``map_location`` so tensors land on the requested device.
        """
        return torch.load(
            os.path.join(graph_dir_path, fn),
            map_location=device,
            weights_only=True,
        )

    # Load static node features
    mesh_static_features = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        [zero_index_edge_index(ei) for ei in loads_file("m2m_edge_index.pt")],
        persistent=False,
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

    # Change first indices to 0
    # m2g and g2m has to be handled specially as not all mesh nodes
    # might be indexed
    m2g_min_indices = m2g_edge_index.min(dim=1, keepdim=True)[0]
    mesh_first = m2g_min_indices[0] < m2g_min_indices[1]
    g2m_edge_index = zero_index_g2m(
        g2m_edge_index, mesh_static_features, mesh_first=mesh_first
    )
    m2g_edge_index = zero_index_m2g(
        m2g_edge_index, mesh_static_features, mesh_first=mesh_first
    )

    assert m2g_edge_index.min() >= 0, "Negative node index in m2g"
    assert g2m_edge_index.min() >= 0, "Negative node index in g2m"

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Not just single level mesh graph

    # Load static edge features
    # List of (M_m2m[l], d_edge_f)
    m2m_features = loads_file("m2m_features.pt")
    g2m_features = loads_file("g2m_features.pt")  # (M_g2m, d_edge_f)
    m2g_features = loads_file("m2g_features.pt")  # (M_m2g, d_edge_f)

    # Normalize by dividing with longest edge (found in m2m)
    longest_edge = max(
        torch.max(level_features[:, 0]) for level_features in m2m_features
    )  # Col. 0 is length

    m2m_features = BufferList(m2m_features, persistent=False)
    m2m_features /= longest_edge
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Some checks for consistency
    assert (
        len(m2m_features) == n_levels
    ), "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            [
                zero_index_edge_index(ei)
                for ei in loads_file("mesh_up_edge_index.pt")
            ],
            persistent=False,
        )  # List of (2, M_up[l])
        mesh_down_edge_index = BufferList(
            [
                zero_index_edge_index(ei)
                for ei in loads_file("mesh_down_edge_index.pt")
            ],
            persistent=False,
        )  # List of (2, M_down[l])

        mesh_up_features = loads_file(
            "mesh_up_features.pt"
        )  # List of (M_up[l], d_edge_f)
        mesh_down_features = loads_file(
            "mesh_down_features.pt"
        )  # List of (M_down[l], d_edge_f)

        # Rescale
        mesh_up_features = BufferList(mesh_up_features, persistent=False)
        mesh_up_features /= longest_edge
        mesh_down_features = BufferList(mesh_down_features, persistent=False)
        mesh_down_features /= longest_edge

        mesh_static_features = BufferList(
            mesh_static_features, persistent=False
        )
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        (
            mesh_up_edge_index,
            mesh_down_edge_index,
            mesh_up_features,
            mesh_down_features,
        ) = ([], [], [], [])

    return hierarchical, {
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_static_features": mesh_static_features,
    }


def make_mlp(blueprint, layer_norm=True):
    """
    Construct a multilayer perceptron from a blueprint of layer widths.

    Parameters
    ----------
    blueprint : list[int]
        Sequence of layer dimensions where ``blueprint[0]`` is the input size,
        ``blueprint[-1]`` is the output size, the intermediate entries specify
        the hidden layer widths, and ``len(blueprint) - 2`` is the number of
        hidden layers.
    layer_norm : bool, optional
        If ``True``, append a ``LayerNorm`` to the output as in GraphCast.

    Returns
    -------
    torch.nn.Sequential
        Sequential module implementing the specified MLP.
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
    Check whether a LaTeX toolchain is available on the system.

    Returns
    -------
    bool
        ``True`` if ``latex`` and the required auxiliary tools are callable.
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
    Return a ``tueplots`` bundle scaled to a fraction of the page width.

    Parameters
    ----------
    fraction : float
        Denominator applied to the default NeurIPS figure width.

    Returns
    -------
    dict
        Matplotlib rcParams bundle with updated ``figure.figsize``.
    """

    usetex = has_working_latex()
    bundle = bundles.neurips2023(usetex=usetex, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] * fraction,
        original_figsize[1] * fraction,
    )
    return bundle


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    """Print arguments only from the rank-zero process in distributed runs."""
    print(*args, **kwargs)


def log_on_rank_zero(msg: str, level: str = "info", *args, **kwargs):
    """Log a message only on rank zero using loguru logger.

    Parameters
    ----------
    msg : str
        The message to log.
    level : str, optional
        The logging level (e.g. "info", "warning", "error"). Default is "info".
    """
    if rank_zero_only.rank == 0:
        log_fn = getattr(logger, level, logger.info)
        log_fn(msg, *args, **kwargs)


def init_training_logger_metrics(training_logger, val_steps):
    """
    Configure validation metric aggregation for the active training logger.

    Parameters
    ----------
    training_logger : pytorch_lightning.loggers.Logger
        Logger instance used during training.
    val_steps : Iterable[int]
        Autoregressive rollout lengths to log as separate metrics.
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
    Instantiate the configured experiment logger.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore providing metadata for logging configuration.
    args : argparse.Namespace
        Parsed training arguments controlling the logger backend.
    run_name : str
        Name of the run.

    Returns
    -------
    training_logger : pytorch_lightning.loggers.base
        Logger object.

    Notes
    -----
    When ``--wandb_id`` is given, ``resume="allow"`` is set automatically:
    W&B resumes the run if it exists, or creates it with that ID otherwise.
    This allows the same job script to be safely resubmitted on HPC systems.
    The run name is set to ``None`` when resuming to preserve the existing name.
    """

    if args.wandb_id and args.logger != "wandb":
        logger.warning(
            f"--wandb_id is set but logger is {args.logger!r}; "
            "the wandb_id will have no effect."
        )

    if args.logger == "wandb":
        wandb_resume = "allow" if args.wandb_id else None
        logger.info(
            f"Wandb resume mode: {wandb_resume!r} (id: {args.wandb_id!r})"
        )
        return pl.loggers.WandbLogger(
            project=args.logger_project,
            name=None if args.wandb_id else run_name,
            config=dict(training=vars(args), datastore=datastore._config),
            resume=wandb_resume,
            id=args.wandb_id,
        )
    elif args.logger == "mlflow":
        if args.wandb_id is not None:
            warnings.warn(
                "--wandb_id is only used with --logger=wandb and will be "
                "ignored."
            )
        url = os.getenv("MLFLOW_TRACKING_URI")
        if url is None:
            raise ValueError(
                "MLFlow logger requires setting MLFLOW_TRACKING_URI in env."
            )
        training_logger = CustomMLFlowLogger(
            experiment_name=args.logger_project,
            tracking_uri=url,
            run_name=run_name,
        )
        training_logger.log_hyperparams(
            dict(training=vars(args), datastore=datastore._config)
        )

    return training_logger


def inverse_softplus(x, beta=1, threshold=20):
    """
    Inverse of :func:`torch.nn.functional.softplus`.

    For most inputs this function is exact up to numerical precision. The
    input is clamped to ensure numerical stability: values above
    ``threshold / beta`` are treated as linear (which is exact in that
    regime), and values near zero are clamped to avoid ``log`` of
    non-positive numbers. Only near the lower clamping bound does the
    result deviate from the true inverse.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor whose softplus inverse should be computed.
    beta : float, optional
        Softplus ``beta`` parameter that controls the sharpness. Default ``1``.
    threshold : float, optional
        Threshold above which the function is treated as linear for numerical
        stability. Default ``20``.

    Returns
    -------
    torch.Tensor
        Tensor containing the inverse-softplus values.

    Notes
    -----
    ``torch.clamp`` will zero the gradients near the bounds, but values this
    close to zero or ``threshold / beta`` already have negligible gradients.
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
    Inverse of ``torch.sigmoid`` with clamping for numerical stability.

    Sigmoid output takes values in ``[0, 1]``; we clamp the input slightly
    within that open interval before applying ``log(x / (1 - x))``.

    Note that ``torch.clamp`` will make gradients 0 near the bounds, but
    this is not a problem as values of x that are this close to 0 or 1
    have gradients of 0 anyhow.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor assumed to contain logits after a sigmoid.

    Returns
    -------
    torch.Tensor
        Tensor containing ``log(x / (1 - x))`` after clamping away from the
        saturation limits.

    Notes
    -----
    ``torch.clamp`` zeroes gradients for values at the bounds, but values this
    close to 0 or 1 already have negligible gradients.
    """
    x_clamped = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    return torch.log(x_clamped / (1 - x_clamped))


def get_integer_time(tdelta) -> tuple[int, str]:
    """
    Express a :class:`datetime.timedelta` as an integer number of time units.

    Parameters
    ----------
    tdelta : datetime.timedelta
        Time interval to convert.

    Returns
    -------
    tuple[int, str]
        Integer value and the corresponding unit (``"weeks"``, ``"days"``,
        ``"hours"``, ``"minutes"``, ``"seconds"``, ``"milliseconds"``, or
        ``"microseconds"``). If no unit yields an integer count,
        ``(1, "unknown")`` is returned.

    Examples
    --------
    >>> from datetime import timedelta
    >>> get_integer_time(timedelta(days=14))
    (2, 'weeks')
    >>> get_integer_time(timedelta(hours=5))
    (5, 'hours')
    >>> get_integer_time(timedelta(milliseconds=1000))
    (1, 'seconds')
    >>> get_integer_time(timedelta(days=0.001))
    (1, 'unknown')
    """
    total_seconds = tdelta.total_seconds()

    units = {
        "weeks": 604800,
        "days": 86400,
        "hours": 3600,
        "minutes": 60,
        "seconds": 1,
        "milliseconds": 0.001,
        "microseconds": 0.000001,
    }

    for unit, unit_in_seconds in units.items():
        if total_seconds % unit_in_seconds == 0:
            return int(total_seconds / unit_in_seconds), unit

    return 1, "unknown"
