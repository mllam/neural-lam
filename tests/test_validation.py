"""
Tests for input validation changes:
- InteractionNet aggregation method
- MLP blueprint validation
- Metric name validation
- CLI argument validation in train_model
"""
import pytest
import torch
from neural_lam.interaction_net import InteractionNet
from neural_lam.utils import make_mlp
from neural_lam.metrics import get_metric
from neural_lam import train_model


def test_invalid_aggregation_method():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        InteractionNet(edge_index, input_dim=4, aggr="max")


def test_invalid_mlp_blueprint():
    with pytest.raises(ValueError, match="Invalid MLP blueprint"):
        make_mlp([4])  # only one dimension, invalid


def test_invalid_metric_name():
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("not_a_metric")


def test_invalid_val_steps():
    args = ["--config_path", "dummy.yaml", "--val_steps_to_log", "20", "--ar_steps_eval", "10"]
    with pytest.raises(ValueError) as excinfo:
        train_model.main(args)
    assert "Can not log validation step 20" in str(excinfo.value)


def test_invalid_var_leads_metrics_watch():
    args = [
        "--config_path", "dummy.yaml",
        "--var_leads_metrics_watch", '{"1":[20]}',
        "--ar_steps_eval", "10"
    ]
    with pytest.raises(ValueError) as excinfo:
        train_model.main(args)
    assert "Can not log validation step 20 for variable 1" in str(excinfo.value)
