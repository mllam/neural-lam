"""Utilities for distributed training configuration."""

# Third-party
import torch


def training_strategy(
    devices: int | str | list[int],
    num_nodes: int,
    *,
    find_unused_parameters: bool = False,
) -> str:
    """Select unused-parameter detection only for distributed training.

    Parameters
    ----------
    devices : int or str or list[int]
        Lightning device count, device indices, or auto selection.
    num_nodes : int
        Number of distributed nodes.
    find_unused_parameters : bool, optional
        Whether the training objective leaves some model parameters unused.

    Returns
    -------
    str
        DDP with unused-parameter detection when requested and distributed;
        otherwise Lightning's auto strategy.
    """
    distributed = (
        num_nodes > 1
        or (isinstance(devices, list) and len(devices) > 1)
        or (isinstance(devices, int) and devices > 1)
        or (devices == "auto" and torch.cuda.device_count() > 1)
    )
    if distributed and find_unused_parameters:
        return "ddp_find_unused_parameters_true"
    return "auto"
