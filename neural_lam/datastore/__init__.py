"""Datastore backends for loading and serving weather model data."""

# Standard library
from pathlib import Path

# Local
from .base import BaseDatastore  # noqa
from .mdp import MDPDatastore  # noqa
from .npyfilesmeps import NpyFilesDatastoreMEPS  # noqa

DATASTORE_CLASSES = [
    MDPDatastore,
    NpyFilesDatastoreMEPS,
]

DATASTORES = {
    datastore.SHORT_NAME: datastore for datastore in DATASTORE_CLASSES
}


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
        return DatastoreClass(config_path=config_path)

    if DatastoreClass is not MDPDatastore:
        raise ValueError(
            "`n_boundary_points` is only supported for "
            f"`kind: {MDPDatastore.SHORT_NAME}` datastores, but was set for "
            f"a `kind: {datastore_kind}` datastore."
        )
    return MDPDatastore(
        config_path=config_path, n_boundary_points=n_boundary_points
    )
