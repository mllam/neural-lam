# Standard library
from unittest.mock import patch

# Third-party
import pytest
import torch
from torch import nn

# First-party
from neural_lam.gnn_layers import InteractionNet
from neural_lam.models.step_predictors.graph.hi_lam import (
    HiLAM,
    SameLevelGNNStack,
)


def _make_edge_index():
    return torch.tensor(
        [[0, 1, 2, 0], [1, 2, 0, 2]],
        dtype=torch.long,
    )


def _make_hi_lam_for_same_level_tests(depths):
    """Build the minimum HiLAM state needed to test same-level GNNs."""
    model = HiLAM.__new__(HiLAM)
    nn.Module.__init__(model)

    edge_index = _make_edge_index()
    model.m2m_edge_index = [edge_index, edge_index, edge_index]
    model.same_level_processor_layers = depths
    model.hidden_dim = 8
    model.hidden_layers = 1

    return model


def test_same_level_gnn_stack_forward_shapes():
    """A multi-layer same-level stack should preserve feature dimensions."""
    edge_index = _make_edge_index()
    model = SameLevelGNNStack(
        edge_index=edge_index,
        num_layers=2,
        hidden_dim=8,
        hidden_layers=1,
    )

    nodes = torch.randn(3, 8)
    edges = torch.randn(edge_index.shape[1], 8)

    output_nodes, output_edges = model(nodes, nodes, edges)

    assert output_nodes.shape == nodes.shape
    assert output_edges.shape == edges.shape


def test_same_level_gnns_use_configured_depths():
    """Each mesh level should receive its configured processor depth."""
    model = _make_hi_lam_for_same_level_tests([1, 2, 3])

    gnns = model.make_same_gnns()

    assert len(gnns) == 3
    assert isinstance(gnns[0], InteractionNet)
    assert isinstance(gnns[1], SameLevelGNNStack)
    assert isinstance(gnns[2], SameLevelGNNStack)
    assert gnns[1].num_layers == 2
    assert gnns[2].num_layers == 3


def test_same_level_default_depth_preserves_interaction_net():
    """Depth one should retain the original InteractionNet module."""
    model = _make_hi_lam_for_same_level_tests([1, 1, 1])

    gnns = model.make_same_gnns()

    assert all(isinstance(gnn, InteractionNet) for gnn in gnns)
    assert all(not isinstance(gnn, SameLevelGNNStack) for gnn in gnns)


@pytest.mark.parametrize(
    "depths",
    [
        [],
        [1, 2],
        [1, 2, 3, 4],
    ],
)
def test_constructor_rejects_wrong_number_of_depths(depths):
    """Reject a depth list that does not match the mesh-level count."""

    def fake_base_init(self, **kwargs):
        nn.Module.__init__(self)
        self.num_levels = 3

    with patch(
        "neural_lam.models.step_predictors.graph.hierarchical."
        "BaseHiGraphModel.__init__",
        fake_base_init,
    ):
        with pytest.raises(ValueError, match="one depth per mesh level"):
            HiLAM(
                datastore=None,
                same_level_processor_layers=depths,
            )


@pytest.mark.parametrize("depth", [0, -1, 1.5, True, "2"])
def test_constructor_rejects_invalid_depths(depth):
    """Reject depths that are not positive integers."""

    def fake_base_init(self, **kwargs):
        nn.Module.__init__(self)
        self.num_levels = 3

    with patch(
        "neural_lam.models.step_predictors.graph.hierarchical."
        "BaseHiGraphModel.__init__",
        fake_base_init,
    ):
        with pytest.raises(
            ValueError,
            match="positive integers",
        ):
            HiLAM(
                datastore=None,
                same_level_processor_layers=[1, depth, 1],
            )


def test_mesh_down_step_supports_multi_layer_same_level_gnns():
    """The down-pass should run with different same-level GNN depths."""
    model = _make_hi_lam_for_same_level_tests([1, 2, 3])
    model.num_levels = 3

    same_gnns = model.make_same_gnns()

    mesh_rep_levels = [
        torch.randn(1, 3, 8),
        torch.randn(1, 3, 8),
        torch.randn(1, 3, 8),
    ]
    mesh_same_rep = [
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
    ]
    mesh_down_rep = [
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
    ]

    class DummyDownGNN(nn.Module):
        """Minimal down-GNN preserving the expected tensor shapes."""

        def forward(self, send_node_rep, rec_node_rep, edge_rep):
            return rec_node_rep, edge_rep

    down_gnns = nn.ModuleList([DummyDownGNN(), DummyDownGNN()])

    updated_nodes, updated_same_edges, updated_down_edges = (
        model.mesh_down_step(
            mesh_rep_levels,
            mesh_same_rep,
            mesh_down_rep,
            down_gnns,
            same_gnns,
        )
    )

    assert len(updated_nodes) == 3
    assert len(updated_same_edges) == 3
    assert len(updated_down_edges) == 2

    for nodes in updated_nodes:
        assert nodes.shape == (1, 3, 8)

    for edges in updated_same_edges:
        assert edges.shape == (1, 4, 8)

    for edges in updated_down_edges:
        assert edges.shape == (1, 4, 8)



def test_same_level_gnn_stack_backpropagates_gradients():
    """All layers in a same-level stack should receive gradients."""
    edge_index = _make_edge_index()
    model = SameLevelGNNStack(
        edge_index=edge_index,
        num_layers=3,
        hidden_dim=8,
        hidden_layers=1,
    )

    nodes = torch.randn(3, 8, requires_grad=True)
    edges = torch.randn(edge_index.shape[1], 8, requires_grad=True)

    output_nodes, output_edges = model(nodes, nodes, edges)
    loss = output_nodes.square().mean() + output_edges.square().mean()
    loss.backward()

    assert nodes.grad is not None
    assert edges.grad is not None

    layers = [
        module
        for module in model.gnn_seq.modules()
        if isinstance(module, InteractionNet)
    ]

    assert len(layers) == 3

    for layer in layers:
        parameters = list(layer.parameters())
        assert parameters
        assert all(parameter.grad is not None for parameter in parameters)


def test_mesh_up_step_supports_multi_layer_same_level_gnns():
    """The up-pass should run with different same-level GNN depths."""
    model = _make_hi_lam_for_same_level_tests([1, 2, 3])

    same_gnns = model.make_same_gnns()

    mesh_rep_levels = [
        torch.randn(1, 3, 8),
        torch.randn(1, 3, 8),
        torch.randn(1, 3, 8),
    ]
    mesh_same_rep = [
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
    ]
    mesh_up_rep = [
        torch.randn(1, 4, 8),
        torch.randn(1, 4, 8),
    ]

    class DummyUpGNN(nn.Module):
        """Minimal up-GNN preserving the expected tensor shapes."""

        def forward(self, send_node_rep, rec_node_rep, edge_rep):
            return rec_node_rep, edge_rep

    up_gnns = nn.ModuleList([DummyUpGNN(), DummyUpGNN()])

    updated_nodes, updated_same_edges, updated_up_edges = (
        model.mesh_up_step(
            mesh_rep_levels,
            mesh_same_rep,
            mesh_up_rep,
            up_gnns,
            same_gnns,
        )
    )

    assert len(updated_nodes) == 3
    assert len(updated_same_edges) == 3
    assert len(updated_up_edges) == 2

    for nodes in updated_nodes:
        assert nodes.shape == (1, 3, 8)

    for edges in updated_same_edges:
        assert edges.shape == (1, 4, 8)

    for edges in updated_up_edges:
        assert edges.shape == (1, 4, 8)

def test_constructor_defaults_to_one_same_level_layer_per_mesh_level():
    """Default configuration should use one same-level layer per mesh level."""

    def fake_base_init(self, **kwargs):
        nn.Module.__init__(self)
        self.num_levels = 3

    with (
        patch(
            "neural_lam.models.step_predictors.graph.hierarchical."
            "BaseHiGraphModel.__init__",
            fake_base_init,
        ),
        patch.object(HiLAM, "make_down_gnns", return_value=nn.ModuleList()),
        patch.object(HiLAM, "make_up_gnns", return_value=nn.ModuleList()),
        patch.object(HiLAM, "make_same_gnns", return_value=nn.ModuleList()),
    ):
        model = HiLAM(datastore=None)

    assert model.same_level_processor_layers == [1, 1, 1]
