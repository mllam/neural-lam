import neural_lam
import neural_lam.create_grid_features
import neural_lam.create_mesh
import neural_lam.create_parameter_weights
import neural_lam.train_model


def test_import():
    assert neural_lam is not None
    assert neural_lam.create_mesh is not None
    assert neural_lam.create_grid_features is not None
    assert neural_lam.create_parameter_weights is not None
    assert neural_lam.train_model is not None
