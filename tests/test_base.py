import neural_lam
import neural_lam.train_model


def test_import():
    assert neural_lam is not None
    assert neural_lam.train_model is not None
