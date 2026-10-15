# Standard library
import subprocess
import sys
from pathlib import Path

# Third-party
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_danra_datastore():
    """Create the DANRA zarr datastore required by hello_world_danra.ipynb."""
    datastore_config = Path(
        "tests/datastore_examples/mdp/danra_100m_winds/danra.datastore.yaml"
    )
    zarr_output = datastore_config.parent / "danra.datastore.zarr"

    # Only create if it doesn't exist
    if not zarr_output.exists():
        subprocess.run(
            [sys.executable, "-m", "mllam_data_prep", str(datastore_config)],
            check=True,
        )
