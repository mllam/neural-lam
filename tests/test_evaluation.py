"""
tests for neural_lam.evaluation verification metrics.
uses small synthetic tensors to check against known analytic results.
"""

# Third-party
import pytest
import torch


class TestComputeGridWeights:
    """grid weight computation"""

    def test_latlon_equator_weight_one(self):
        """equator should get highest weight since cos(0)=1"""
        # First-party
        from neural_lam.evaluation import compute_grid_weights

        coords = torch.tensor(
            [
                [0.0, 0.0],  # equator
                [0.0, 30.0],
                [0.0, 60.0],
                [0.0, 90.0],  # pole
            ]
        )
        w = compute_grid_weights(coords, grid_type="latlon")
        assert w[0] > w[1] > w[2] > w[3]
        # should sum to N=4
        assert abs(w.sum().item() - 4.0) < 1e-5

    def test_equal_area_uniform(self):
        """equal-area grids just return ones"""
        # First-party
        from neural_lam.evaluation import compute_grid_weights

        coords = torch.randn(100, 2)
        w = compute_grid_weights(coords, grid_type="equal_area")
        assert torch.allclose(w, torch.ones(100))

    def test_invalid_grid_type(self):
        # First-party
        from neural_lam.evaluation import compute_grid_weights

        with pytest.raises(ValueError, match="Unknown grid_type"):
            compute_grid_weights(torch.randn(10, 2), grid_type="invalid")


class TestWeightedRMSE:
    """weighted rmse and mae"""

    def test_perfect_prediction(self):
        """rmse should be 0 when pred == target"""
        # First-party
        from neural_lam.evaluation import weighted_rmse

        pred = torch.randn(4, 10, 3)
        result = weighted_rmse(pred, pred)
        assert torch.allclose(result, torch.zeros(4, 3), atol=1e-6)

    def test_known_rmse(self):
        """constant error of 2 everywhere -> rmse = 2"""
        # First-party
        from neural_lam.evaluation import weighted_rmse

        pred = torch.ones(2, 8, 1) * 3.0
        target = torch.ones(2, 8, 1) * 1.0
        result = weighted_rmse(pred, target)
        assert torch.allclose(result, torch.tensor([[2.0], [2.0]]))

    def test_weighted_vs_unweighted(self):
        """upweighting the bigger-error node should increase rmse"""
        # First-party
        from neural_lam.evaluation import weighted_rmse

        pred = torch.tensor([[[1.0], [3.0]]])
        target = torch.tensor([[[0.0], [0.0]]])
        unweighted = weighted_rmse(pred, target)

        weights = torch.tensor([0.1, 0.9])
        weighted = weighted_rmse(pred, target, grid_weights=weights)
        assert weighted > unweighted

    def test_mae_perfect(self):
        # First-party
        from neural_lam.evaluation import weighted_mae

        pred = torch.randn(2, 10, 5)
        result = weighted_mae(pred, pred)
        assert torch.allclose(result, torch.zeros(2, 5), atol=1e-6)

    def test_mask_excludes_nodes(self):
        """masking out node 1 should ignore its error"""
        # First-party
        from neural_lam.evaluation import weighted_rmse

        pred = torch.tensor([[[0.0], [100.0]]])
        target = torch.zeros(1, 2, 1)
        mask = torch.tensor([True, False])
        result = weighted_rmse(pred, target, mask=mask)
        assert torch.allclose(result, torch.zeros(1, 1), atol=1e-6)


class TestLatitudeWeightedRMSE:

    def test_perfect_prediction(self):
        # First-party
        from neural_lam.evaluation import latitude_weighted_rmse

        pred = torch.randn(2, 10, 3)
        coords = torch.randn(10, 2) * 45
        result = latitude_weighted_rmse(pred, pred, coords)
        assert torch.allclose(result, torch.zeros(2, 3), atol=1e-6)

    def test_equator_vs_pole_emphasis(self):
        """error at equator should matter more than at pole"""
        # First-party
        from neural_lam.evaluation import latitude_weighted_rmse

        coords = torch.tensor([[0.0, 0.0], [0.0, 89.0]])

        pred_a = torch.tensor([[[1.0], [0.0]]])  # error at equator
        pred_b = torch.tensor([[[0.0], [1.0]]])  # error at pole
        target = torch.zeros(1, 2, 1)

        rmse_a = latitude_weighted_rmse(pred_a, target, coords)
        rmse_b = latitude_weighted_rmse(pred_b, target, coords)
        assert rmse_a > rmse_b


class TestACC:
    """anomaly correlation coefficient"""

    def test_perfect_forecast(self):
        """pred == target should give acc = 1"""
        # First-party
        from neural_lam.evaluation import acc

        target = torch.randn(4, 20, 3)
        clim = torch.randn(20, 3)
        result = acc(target, target, clim)
        assert torch.allclose(result, torch.ones(4, 3), atol=1e-5)

    def test_climatology_forecast(self):
        """predicting climatology should give acc = 0"""
        # First-party
        from neural_lam.evaluation import acc

        target = torch.randn(4, 20, 3)
        clim = torch.randn(20, 3)
        result = acc(clim.unsqueeze(0).expand_as(target), target, clim)
        assert torch.allclose(result, torch.zeros(4, 3), atol=1e-5)

    def test_opposite_anomalies(self):
        """opposite anomaly signs -> acc = -1"""
        # First-party
        from neural_lam.evaluation import acc

        clim = torch.zeros(10, 1)
        target = torch.ones(1, 10, 1)
        pred = -torch.ones(1, 10, 1)
        result = acc(pred, target, clim)
        assert torch.allclose(
            result, torch.tensor([[[-1.0]]]).squeeze(0), atol=1e-5
        )

    def test_acc_range(self):
        """should always be in [-1, 1]"""
        # First-party
        from neural_lam.evaluation import acc

        pred = torch.randn(8, 50, 5)
        target = torch.randn(8, 50, 5)
        clim = torch.randn(50, 5)
        result = acc(pred, target, clim)
        assert (result >= -1.0 - 1e-5).all() and (result <= 1.0 + 1e-5).all()

    def test_weighted_acc(self):
        # First-party
        from neural_lam.evaluation import acc

        pred = torch.randn(2, 10, 3)
        target = torch.randn(2, 10, 3)
        clim = torch.randn(10, 3)
        weights = torch.rand(10)
        result = acc(pred, target, clim, grid_weights=weights)
        assert result.shape == (2, 3)


class TestSpreadSkillRatio:

    def test_perfect_calibration(self):
        """spread == skill -> ratio 1.0"""
        # First-party
        from neural_lam.evaluation import spread_skill_ratio

        pred = torch.zeros(4, 10, 3)
        target = torch.ones(4, 10, 3)
        pred_std = torch.ones(4, 10, 3)
        result = spread_skill_ratio(pred, target, pred_std)
        assert torch.allclose(result, torch.ones(4, 3), atol=1e-5)

    def test_overconfident(self):
        """tiny std + big error -> ratio < 1"""
        # First-party
        from neural_lam.evaluation import spread_skill_ratio

        pred = torch.zeros(2, 10, 1)
        target = torch.ones(2, 10, 1) * 5.0
        pred_std = torch.ones(2, 10, 1) * 0.1
        result = spread_skill_ratio(pred, target, pred_std)
        assert (result < 1.0).all()

    def test_overdispersive(self):
        """huge std + small error -> ratio > 1"""
        # First-party
        from neural_lam.evaluation import spread_skill_ratio

        pred = torch.zeros(2, 10, 1)
        target = torch.ones(2, 10, 1) * 0.01
        pred_std = torch.ones(2, 10, 1) * 10.0
        result = spread_skill_ratio(pred, target, pred_std)
        assert (result > 1.0).all()

    def test_shape_output(self):
        # First-party
        from neural_lam.evaluation import spread_skill_ratio

        result = spread_skill_ratio(
            torch.randn(3, 5, 20, 4),
            torch.randn(3, 5, 20, 4),
            torch.ones(3, 5, 20, 4),
        )
        assert result.shape == (3, 5, 4)
