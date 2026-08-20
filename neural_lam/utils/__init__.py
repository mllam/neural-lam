"""Utility helpers shared across Neural-LAM training and evaluation."""

# Local
from .buffer_list import BufferList
from .graph import (
    compute_grid_input_dim,
    load_and_register_graph,
    load_graph,
    zero_index_edge_index,
    zero_index_g2m,
    zero_index_m2g,
)
from .logging import (
    init_training_logger_metrics,
    log_on_rank_zero,
    setup_training_logger,
)
from .networks import make_gnn_seq, make_mlp
from .plot import fractional_plot_bundle, has_working_latex
from .tensor import inverse_sigmoid, inverse_softplus
from .time import (
    apply_time_crop,
    check_time_overlap,
    crop_time_if_needed,
    get_integer_time,
    get_time_crop_slice,
    get_time_step,
)

__all__ = [
    "BufferList",
    "apply_time_crop",
    "check_time_overlap",
    "compute_grid_input_dim",
    "crop_time_if_needed",
    "fractional_plot_bundle",
    "get_integer_time",
    "get_time_crop_slice",
    "get_time_step",
    "has_working_latex",
    "init_training_logger_metrics",
    "inverse_sigmoid",
    "inverse_softplus",
    "load_and_register_graph",
    "load_graph",
    "log_on_rank_zero",
    "make_gnn_seq",
    "make_mlp",
    "setup_training_logger",
    "zero_index_edge_index",
    "zero_index_g2m",
    "zero_index_m2g",
]
