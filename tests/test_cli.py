# First-party
import neural_lam
import neural_lam.create_graph
import neural_lam.train_model


def test_import():
    """This test just ensures that each cli entry-point can be imported for now,
    eventually we should test their execution too."""
    assert neural_lam is not None
    assert neural_lam.create_graph is not None
    assert neural_lam.train_model is not None
