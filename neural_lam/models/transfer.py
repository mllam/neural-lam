"""Selective initialization of model components from checkpoints."""

# Standard library
from collections.abc import Mapping, Sequence
from pathlib import Path

# Third-party
import torch

# Local
from .step_predictors.graph.graph_lam import GraphLAM

SUPPORTED_GRAPH_LAM_TRANSFER_MODULES = frozenset({"processor"})


def initialize_graph_lam_from_checkpoint(
    target_model: GraphLAM,
    checkpoint_path: str | Path,
    modules: Sequence[str],
) -> None:
    """Initialize selected GraphLAM modules from a Lightning checkpoint.

    The complete state of every selected module must be present and have the
    same parameter shapes as the target module. Validation finishes before any
    target weights are changed.

    Parameters
    ----------
    target_model : GraphLAM
        Newly initialized model that receives the pretrained weights.
    checkpoint_path : str or pathlib.Path
        Path to a trusted Neural-LAM Lightning checkpoint.
    modules : sequence of str
        GraphLAM modules to initialize. Currently only ``"processor"`` is
        supported.

    Raises
    ------
    TypeError
        If ``target_model`` is not a GraphLAM model.
    ValueError
        If the module selection or checkpoint is incompatible with the target
        model.
    """
    if not isinstance(target_model, GraphLAM):
        raise TypeError("Selective initialization currently requires GraphLAM.")

    selected_modules = tuple(modules)
    if not selected_modules:
        raise ValueError("At least one GraphLAM module must be selected.")
    if len(set(selected_modules)) != len(selected_modules):
        raise ValueError(
            "GraphLAM transfer modules must not contain duplicates."
        )

    unsupported = sorted(
        set(selected_modules) - SUPPORTED_GRAPH_LAM_TRANSFER_MODULES
    )
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_GRAPH_LAM_TRANSFER_MODULES))
        raise ValueError(
            f"Unsupported GraphLAM transfer modules: {unsupported}. "
            f"Supported modules: {supported}."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping.")

    source_state = checkpoint.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise ValueError("Checkpoint does not contain a valid 'state_dict'.")

    validated_states: dict[str, dict[str, torch.Tensor]] = {}
    for module_name in selected_modules:
        prefix = f"forecaster.predictor.{module_name}."
        raw_module_state = {
            key.removeprefix(prefix): value
            for key, value in source_state.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if not raw_module_state:
            raise ValueError(
                f"Checkpoint does not contain GraphLAM module {module_name!r}."
            )

        target_state = getattr(target_model, module_name).state_dict()
        missing = sorted(target_state.keys() - raw_module_state.keys())
        unexpected = sorted(raw_module_state.keys() - target_state.keys())
        if missing or unexpected:
            raise ValueError(
                f"Incompatible state keys for GraphLAM module {module_name!r}: "
                f"missing={missing}, unexpected={unexpected}."
            )

        non_tensors = sorted(
            key
            for key, value in raw_module_state.items()
            if not isinstance(value, torch.Tensor)
        )
        if non_tensors:
            raise ValueError(
                f"Non-tensor state values for GraphLAM module {module_name!r}: "
                f"{non_tensors}."
            )

        module_state = {
            key: value
            for key, value in raw_module_state.items()
            if isinstance(value, torch.Tensor)
        }
        shape_mismatches = [
            f"{key}: checkpoint {tuple(module_state[key].shape)} != "
            f"target {tuple(target_state[key].shape)}"
            for key in target_state
            if module_state[key].shape != target_state[key].shape
        ]
        if shape_mismatches:
            raise ValueError(
                f"Incompatible parameter shapes for GraphLAM module "
                f"{module_name!r}: {'; '.join(shape_mismatches)}."
            )

        validated_states[module_name] = module_state

    for module_name, module_state in validated_states.items():
        getattr(target_model, module_name).load_state_dict(
            module_state, strict=True
        )
