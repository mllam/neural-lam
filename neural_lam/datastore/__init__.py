# Local
from .mllam import MLLAMDatastore  # noqa
from .npyfiles import NpyFilesDatastore  # noqa

DATASTORES = dict(
    mllam=MLLAMDatastore,
    npyfiles=NpyFilesDatastore,
)


def init_datastore(datastore_kind, config_path):
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    datastore = DatastoreClass(config_path=config_path)

    return datastore
