# Local
from .base import BaseDatastore  # noqa
from .mdp import MDPDatastore  # noqa
from .npyfilesmeps import NpyFilesDatastoreMEPS  # noqa

DATASTORE_CLASSES = [
    MDPDatastore,
    NpyFilesDatastoreMEPS,
]

DATASTORES = {
    datastore.SHORT_NAME: datastore  # type: ignore
    for datastore in DATASTORE_CLASSES
}


def init_datastore(datastore_kind, config_path):
    """
    Instantiate a datastore based on its short-name identifier.

    Parameters
    ----------
    datastore_kind : str
        Key corresponding to one of :data:`DATASTORES`.
    config_path : str | pathlib.Path
        Path to the datastore-specific configuration file.

    Returns
    -------
    BaseDatastore
        Concrete datastore instance configured for ``config_path``.

    Raises
    ------
    NotImplementedError
        If ``datastore_kind`` is not registered.
    """
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    datastore = DatastoreClass(config_path=config_path)

    return datastore
