"""Datastore backends for loading and serving weather model data."""

# Standard library
from pathlib import Path
from typing import Callable, TypeVar

# Local
from .base import BaseDatastore  # noqa: F401
from .mdp import MDPDatastore  # noqa: F401
from .npyfilesmeps import NpyFilesDatastoreMEPS  # noqa: F401

T = TypeVar("T", bound=type[BaseDatastore])

DATASTORES: dict[str, type[BaseDatastore]] = {}


def register_datastore(name: str) -> Callable[[T], T]:
    """
    Decorator to register a BaseDatastore class in the DATASTORES registry.

    Parameters
    ----------
    name : str
        The short name identifier for the datastore backend.

    Returns
    -------
    Callable
        Decorator registering the datastore class under ``name``.
    """

    def decorator(cls: T) -> T:
        """Register the datastore class."""
        if name in DATASTORES:
            raise ValueError(
                f"Datastore '{name}' is already registered to "
                f"{DATASTORES[name]}"
            )
        DATASTORES[name] = cls
        return cls

    return decorator


register_datastore(MDPDatastore.SHORT_NAME)(MDPDatastore)
register_datastore(NpyFilesDatastoreMEPS.SHORT_NAME)(NpyFilesDatastoreMEPS)


def init_datastore(
    datastore_kind: str,
    config_path: str | Path,
    n_boundary_points: int | None = None,
) -> BaseDatastore:
    """
    Instantiate a datastore based on its short-name identifier.

    Parameters
    ----------
    datastore_kind : str
        Key corresponding to one of :data:`DATASTORES`.
    config_path : str | pathlib.Path
        Path to the datastore-specific configuration file.
    n_boundary_points : int, optional
        Forwarded to :class:`~neural_lam.datastore.mdp.MDPDatastore` when
        ``datastore_kind`` is ``"mdp"``, overriding its default boundary-mask
        width. Must be left unset for other datastore kinds.

    Returns
    -------
    BaseDatastore
        Concrete datastore instance configured for ``config_path``.

    Raises
    ------
    NotImplementedError
        If ``datastore_kind`` is not registered.
    ValueError
        If ``n_boundary_points`` is set for a ``datastore_kind`` other than
        :data:`MDPDatastore.SHORT_NAME`.
    """
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    if n_boundary_points is None:
        datastore: BaseDatastore = DatastoreClass(
            config_path=config_path  # ty: ignore[unknown-argument]
        )
        return datastore

    if DatastoreClass is not MDPDatastore:
        raise ValueError(
            "`n_boundary_points` is only supported for "
            f"`kind: {MDPDatastore.SHORT_NAME}` datastores, but was set for "
            f"a `kind: {datastore_kind}` datastore."
        )
    return MDPDatastore(
        config_path=config_path, n_boundary_points=n_boundary_points
    )
