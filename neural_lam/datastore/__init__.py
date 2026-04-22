# Local
from .base import BaseDatastore  # noqa: F401
from .mdp import MDPDatastore  # noqa: F401
from .npyfilesmeps import NpyFilesDatastoreMEPS  # noqa: F401

DATASTORE_CLASSES = [
    MDPDatastore,
    NpyFilesDatastoreMEPS,
]

DATASTORES = {
    datastore.SHORT_NAME: datastore  # type: ignore
    for datastore in DATASTORE_CLASSES
}


def init_datastore(datastore_kind, config_path):
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    datastore = DatastoreClass(config_path=config_path)

    return datastore
