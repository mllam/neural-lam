# Local
from .mdp import MDPDatastore  # noqa
from .npyfilesmeps import NpyFilesDatastoreMEPS  # noqa

DATASTORES = dict(
    mdp=MDPDatastore,
    npyfilesmeps=NpyFilesDatastoreMEPS,
)


def init_datastore(datastore_kind, config_path):
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    datastore = DatastoreClass(config_path=config_path)

    return datastore
