# Third-party
import pytest

# First-party
from neural_lam.datastore.mdp import MDPDatastore

STATE_ONLY_CONFIG = (
    "tests/datastore_examples/mdp/danra_100m_winds/state_only.datastore.yaml"
)


def test_state_only_datastore_no_static_warning():
    """Test that a datastore with no static features emits a warning
    instead of raising an error."""
    with pytest.warns(UserWarning, match="static"):
        datastore = MDPDatastore(config_path=STATE_ONLY_CONFIG)
        assert datastore is not None
