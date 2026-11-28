"""Registry of checkpoint state-dict migrations.

Handles the case where a checkpoint was saved by an older version of
neural-lam whose model classes have since been renamed, restructured or
removed. See issue #48 for the background discussion.

Checkpoints are stamped with an internal, monotonically increasing
``neural_lam_checkpoint_version`` (unrelated to the package's own semver,
since that is derived from git tags and is not meaningful for dev
installs). Each entry in ``MIGRATIONS`` describes how to bring a state
dict from the version before it up to its ``target_version``.
"""

# Standard library
from dataclasses import dataclass
from typing import Callable, Optional

StateDict = dict


@dataclass(frozen=True)
class Migration:
    """A single step that brings a state dict up to ``target_version``."""

    target_version: int
    description: str
    apply: Optional[Callable[[StateDict], StateDict]] = None
    hard_break: bool = False

    def __post_init__(self) -> None:
        """Validate that exactly one of `apply` or `hard_break` is set."""
        if self.hard_break == (self.apply is not None):
            raise ValueError(
                "Migration must set exactly one of `apply` or `hard_break`"
            )


def rename_keys(
    old_prefix: str, new_prefix: str
) -> Callable[[StateDict], StateDict]:
    """
    Build a migration function that renames all state dict keys starting
    with ``old_prefix`` to start with ``new_prefix`` instead.
    """

    def _apply(state_dict: StateDict) -> StateDict:
        """Rename keys starting with `old_prefix` to start with `new_prefix`."""
        for old_key in [
            key for key in state_dict if key.startswith(old_prefix)
        ]:
            new_key = new_prefix + old_key[len(old_prefix) :]
            state_dict[new_key] = state_dict.pop(old_key)
        return state_dict

    return _apply


def remove_keys(keys: list[str]) -> Callable[[StateDict], StateDict]:
    """Build a migration function that drops the given keys, if present."""

    def _apply(state_dict: StateDict) -> StateDict:
        """Drop `keys` from `state_dict`, if present."""
        for key in keys:
            state_dict.pop(key, None)
        return state_dict

    return _apply


def _prefix_flat_ar_model_keys(state_dict: StateDict) -> StateDict:
    """
    Move every parameter of the pre-refactor, flat ``ARModel`` under
    ``forecaster.predictor.``, matching the ``ForecasterModule`` /
    ``Forecaster`` / ``StepPredictor`` split introduced in #208.
    Non-persistent buffers (``interior_mask_bool`` and ``per_var_std``)
    are dropped since modern modules register them with persistent=False.
    """
    state_dict.pop("interior_mask_bool", None)
    state_dict.pop("per_var_std", None)
    for old_key in [
        key for key in state_dict if not key.startswith("forecaster.")
    ]:
        state_dict[f"forecaster.predictor.{old_key}"] = state_dict.pop(old_key)
    return state_dict


def _rename_grid_mlp_keys(state_dict: StateDict) -> StateDict:
    """Rename 'g2m_gnn.grid_mlp' to 'encoding_grid_mlp'."""
    for old_key in [key for key in state_dict if "g2m_gnn.grid_mlp" in key]:
        new_key = old_key.replace("g2m_gnn.grid_mlp", "encoding_grid_mlp")
        state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


# Ordered oldest to newest. Each migration is applied to a state dict that
# is already at `target_version - 1`ish (in practice: at any version below
# `target_version`), so migrations must remain valid to run in sequence.
MIGRATIONS: list[Migration] = [
    Migration(
        target_version=1,
        description=(
            "Rename 'g2m_gnn.grid_mlp' to 'encoding_grid_mlp', from moving "
            "the grid MLP out of the InteractionNet class."
        ),
        apply=_rename_grid_mlp_keys,
    ),
    Migration(
        target_version=2,
        description=(
            "Prefix flat ARModel parameters with 'forecaster.predictor.', "
            "from splitting ARModel into ForecasterModule/Forecaster/"
            "StepPredictor (#208)."
        ),
        apply=_prefix_flat_ar_model_keys,
    ),
]

CURRENT_CHECKPOINT_VERSION = max(
    (migration.target_version for migration in MIGRATIONS), default=0
)


# Schema versions for unversioned legacy checkpoints
LEGACY_FLAT_SCHEMA_VERSION = 0  # Pre-#208 flat ARModel (v0.6.0 and earlier)
V0_7_0_MODULAR_SCHEMA_VERSION = 2  # Post-#208 modular layout (v0.7.0 era)


def detect_checkpoint_version(checkpoint: dict) -> int:
    """
    Determine the schema version of a checkpoint.

    If the checkpoint has an explicit version tag, return it. Otherwise,
    inspect the state dict keys to infer whether the checkpoint predates
    the model hierarchy split (#208) or was saved with the modular layout
    (e.g., release tag v0.7.0).

    Parameters
    ----------
    checkpoint : dict
        The loaded checkpoint dictionary.

    Returns
    -------
    int
        The detected checkpoint version.
    """
    if "neural_lam_checkpoint_version" in checkpoint:
        return int(checkpoint["neural_lam_checkpoint_version"])

    state_dict = checkpoint.get("state_dict", {})
    if any(k.startswith("forecaster.") for k in state_dict):
        # Modern modular layout (v0.7.0 era, post-#208 and grid_mlp rename)
        return V0_7_0_MODULAR_SCHEMA_VERSION

    # Pre-#208 flat ARModel layout (v0.6.0 or earlier)
    return LEGACY_FLAT_SCHEMA_VERSION


def apply_migrations(
    state_dict: StateDict, from_version: int
) -> tuple[StateDict, int]:
    """
    Migrate ``state_dict`` from ``from_version`` up to
    ``CURRENT_CHECKPOINT_VERSION``, applying each intervening migration in
    order.

    Parameters
    ----------
    state_dict : dict
        The checkpoint's state dict, migrated in place and also returned.
    from_version : int
        The checkpoint's ``neural_lam_checkpoint_version`` (0 if the
        checkpoint predates this versioning scheme entirely).

    Returns
    -------
    tuple[dict, int]
        The migrated state dict, and the version it now conforms to. Equal
        to ``CURRENT_CHECKPOINT_VERSION`` unless a hard break stopped the
        chain early (which raises rather than returning).

    Raises
    ------
    RuntimeError
        If a migration in the chain is a hard break, i.e. one that cannot
        convert old parameters into new ones automatically.
    """
    reached_version = from_version
    for migration in sorted(MIGRATIONS, key=lambda m: m.target_version):
        if migration.target_version <= reached_version:
            continue
        if migration.hard_break:
            raise RuntimeError(
                f"Checkpoint at version {reached_version} cannot be "
                f"automatically migrated past version "
                f"{migration.target_version}: {migration.description}. "
                "Please re-train, or convert it using the neural-lam "
                "version just before this change."
            )
        assert migration.apply is not None  # guaranteed by __post_init__
        state_dict = migration.apply(state_dict)
        reached_version = migration.target_version
    return state_dict, reached_version
