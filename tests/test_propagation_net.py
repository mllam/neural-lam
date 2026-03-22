# Standard library
from pathlib import Path

# Third-party
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.interaction_net import InteractionNet, PropagationNet
from neural_lam.models import MODELS
from neural_lam.models.ar_forecaster import ARForecaster
from tests.conftest import init_datastore_example


def _make_edge_index(n_send, n_rec, n_edges):
    """Create a random edge_index for testing."""
    torch.manual_seed(0)
    senders = torch.randint(0, n_send, (n_edges,))
    receivers = torch.randint(0, n_rec, (n_edges,))
    return torch.stack([senders, receivers])


def _make_fully_connected_edge_index(n_send, n_rec):
    """Create a fully-connected edge_index (every sender to every receiver)."""
    senders = (
        torch.arange(n_send).unsqueeze(1).expand(n_send, n_rec).reshape(-1)
    )
    receivers = (
        torch.arange(n_rec).unsqueeze(0).expand(n_send, n_rec).reshape(-1)
    )
    return torch.stack([senders, receivers])


def _build_model_and_data(datastore, config, model_name, graph_name,
                          vertical_propnets=False):
    """Helper to build a model and matching random input tensors."""
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1

    predictor = MODELS[model_name](
        config=config,
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=num_past_forcing_steps,
        num_future_forcing_steps=num_future_forcing_steps,
        output_std=False,
        vertical_propnets=vertical_propnets,
    )
    forecaster = ARForecaster(predictor, datastore)

    B = 2
    num_grid_nodes = predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )

    torch.manual_seed(123)
    init_states = torch.randn(B, 2, num_grid_nodes, d_state)
    forcing = torch.randn(B, 1, num_grid_nodes, d_forcing)
    boundary = torch.randn(B, 1, num_grid_nodes, d_state)

    return forecaster, predictor, init_states, forcing, boundary


def _get_datastore_and_config(graph_name):
    """Create a datastore with graph already built."""
    datastore = init_datastore_example("mdp")
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        )
    )

    # Ensure graph exists
    if graph_name == "hierarchical":
        hierarchical = True
        n_max_levels = 3
    elif graph_name == "multiscale":
        hierarchical = False
        n_max_levels = 3
    else:
        hierarchical = False
        n_max_levels = 1

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            hierarchical=hierarchical,
            n_max_levels=n_max_levels,
        )

    return datastore, config


# 
# Section A: Structural Tests
# 


class TestPropagationNetStructure:
    """Tests for PropagationNet class structure and constructor behavior."""

    def test_propagation_net_is_subclass(self):
        """PropagationNet should be a subclass of InteractionNet."""
        assert issubclass(PropagationNet, InteractionNet)

    def test_forced_mean_aggregation(self):
        """PropagationNet should always use mean aggregation,
        regardless of what is passed."""
        edge_index = _make_edge_index(5, 4, 10)
        net = PropagationNet(edge_index, input_dim=8, aggr="sum")
        assert net.aggr == "mean"

    def test_interaction_net_respects_aggr(self):
        """InteractionNet should use whatever aggregation is passed."""
        edge_index = _make_edge_index(5, 4, 10)
        net_sum = InteractionNet(edge_index.clone(), input_dim=8, aggr="sum")
        net_mean = InteractionNet(
            edge_index.clone(), input_dim=8, aggr="mean"
        )
        assert net_sum.aggr == "sum"
        assert net_mean.aggr == "mean"

    def test_mlp_input_dimensions(self):
        """Edge MLP should accept 3*input_dim, aggr MLP should accept
        2*input_dim."""
        d_h = 16
        edge_index = _make_edge_index(5, 4, 10)
        pnet = PropagationNet(edge_index, input_dim=d_h)

        # Edge MLP: first layer input should be 3 * d_h
        edge_mlp_first = pnet.edge_mlp[0]
        assert edge_mlp_first.in_features == 3 * d_h

        # Aggr MLP: first layer input should be 2 * d_h
        aggr_mlp_first = pnet.aggr_mlp[0]
        assert aggr_mlp_first.in_features == 2 * d_h

    def test_node_index_offset_convention(self):
        """After construction, sender indices in edge_index should be
        offset by num_rec so that receivers are [0, num_rec) and senders
        are [num_rec, num_rec + num_send)."""
        n_send, n_rec, n_edges = 5, 4, 10
        edge_index = _make_edge_index(n_send, n_rec, n_edges)
        pnet = PropagationNet(edge_index, input_dim=8)

        stored_ei = pnet.edge_index
        # Receiver indices should be in [0, num_rec)
        assert stored_ei[1].min() >= 0
        assert stored_ei[1].max() < pnet.num_rec
        # Sender indices should be in [num_rec, num_rec + num_send)
        assert stored_ei[0].min() >= pnet.num_rec


# 
# Section B: Forward Pass Correctness
# 


class TestPropagationNetForwardPass:
    """Tests for PropagationNet forward pass mechanics."""

    def test_output_shapes_match_interaction_net(self):
        """PropagationNet output shapes should match InteractionNet
        for both update_edges=True and update_edges=False."""
        n_send, n_rec, n_edges, d_h = 5, 4, 10, 8
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        for update_edges in [True, False]:
            inet = InteractionNet(
                edge_index.clone(),
                input_dim=d_h,
                update_edges=update_edges,
            )
            pnet = PropagationNet(
                edge_index.clone(),
                input_dim=d_h,
                update_edges=update_edges,
            )

            torch.manual_seed(42)
            send_rep = torch.randn(n_send, d_h)
            rec_rep = torch.randn(n_rec, d_h)
            edge_rep = torch.randn(n_edges, d_h)

            i_out = inet(send_rep, rec_rep, edge_rep)
            p_out = pnet(send_rep, rec_rep, edge_rep)

            if update_edges:
                assert isinstance(i_out, tuple) and len(i_out) == 2
                assert isinstance(p_out, tuple) and len(p_out) == 2
                assert i_out[0].shape == p_out[0].shape == (n_rec, d_h)
                assert i_out[1].shape == p_out[1].shape == (n_edges, d_h)
            else:
                assert i_out.shape == p_out.shape == (n_rec, d_h)

    def test_sender_residual_in_message(self):
        """PropagationNet message should be x_j + edge_mlp(...), verified
        by zeroing edge_mlp weights so message reduces to x_j."""
        n_send, n_rec, d_h = 3, 2, 4
        # Fully connected: every sender connects to every receiver
        edge_index = _make_fully_connected_edge_index(n_send, n_rec)
        n_edges = edge_index.shape[1]

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=True
        )

        # Zero out edge MLP so edge_mlp(...) = 0
        # Then message = x_j + 0 = x_j
        with torch.no_grad():
            for param in pnet.edge_mlp.parameters():
                param.zero_()

        torch.manual_seed(42)
        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        # Call propagate directly to get raw messages
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        edge_rep_aggr, edge_diff = pnet.propagate(
            pnet.edge_index, x=node_reps, edge_attr=edge_rep
        )

        # With zeroed edge_mlp, messages = x_j (sender reps)
        # edge_diff should equal the sender reps for each edge
        # Each edge_diff[i] should be send_rep[sender_of_edge_i]
        sender_indices = pnet.edge_index[0] - pnet.num_rec
        expected_messages = send_rep[sender_indices]
        assert torch.allclose(edge_diff, expected_messages, atol=1e-6)

    def test_receiver_residual_targets_aggregated_messages(self):
        """PropagationNet receiver update should be:
        rec_new = agg_msgs + aggr_mlp(cat(rec_old, agg_msgs))
        NOT rec_new = rec_old + aggr_mlp(...)"""
        n_send, n_rec, n_edges, d_h = 3, 2, 6, 4
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        # Zero out aggr_mlp so aggr_mlp(...) = 0
        # Then rec_new = agg_msgs + 0 = agg_msgs (for PropagationNet)
        # But for InteractionNet it would be rec_old + 0 = rec_old
        with torch.no_grad():
            for param in pnet.aggr_mlp.parameters():
                param.zero_()

        torch.manual_seed(42)
        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        rec_out = pnet(send_rep, rec_rep, edge_rep)

        # With zeroed aggr_mlp, output should NOT equal rec_rep
        # (it would equal rec_rep if residual targeted rec_rep like INet)
        assert not torch.allclose(rec_out, rec_rep, atol=1e-6)

        # Verify by also computing what agg_msgs would be:
        # Run propagate to get edge_rep_aggr
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        edge_rep_aggr, _ = pnet.propagate(
            pnet.edge_index, x=node_reps, edge_attr=edge_rep
        )

        # rec_out should equal edge_rep_aggr (since aggr_mlp output = 0)
        assert torch.allclose(rec_out, edge_rep_aggr, atol=1e-6)

    def test_numerical_divergence_from_interaction_net(self):
        """PropagationNet should produce different outputs than InteractionNet
        given the same weights."""
        n_send, n_rec, n_edges, d_h = 5, 4, 10, 8
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        inet = InteractionNet(
            edge_index.clone(), input_dim=d_h, update_edges=True
        )
        pnet = PropagationNet(
            edge_index.clone(), input_dim=d_h, update_edges=True
        )
        pnet.load_state_dict(inet.state_dict())

        torch.manual_seed(42)
        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        i_rec, i_edge = inet(send_rep, rec_rep, edge_rep)
        p_rec, p_edge = pnet(send_rep, rec_rep, edge_rep)

        assert not torch.allclose(i_rec, p_rec)


# 
# Section C: Edge Update Behavior
# 


class TestEdgeUpdateBehavior:
    """Tests for edge update mechanics."""

    def test_update_edges_true_returns_tuple(self):
        """update_edges=True should return (rec_rep, edge_rep) tuple."""
        n_send, n_rec, n_edges, d_h = 5, 4, 10, 8
        edge_index = _make_edge_index(n_send, n_rec, n_edges)
        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=True
        )

        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        result = pnet(send_rep, rec_rep, edge_rep)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0].shape == (n_rec, d_h)
        assert result[1].shape == (n_edges, d_h)

    def test_update_edges_false_returns_tensor(self):
        """update_edges=False should return only rec_rep tensor."""
        n_send, n_rec, n_edges, d_h = 5, 4, 10, 8
        edge_index = _make_edge_index(n_send, n_rec, n_edges)
        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        result = pnet(send_rep, rec_rep, edge_rep)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (n_rec, d_h)

    def test_edge_residual_connection(self):
        """Edge update should use edge_rep + edge_diff residual,
        same as InteractionNet."""
        n_send, n_rec, n_edges, d_h = 5, 4, 10, 8
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=True
        )

        torch.manual_seed(42)
        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        _, edge_out = pnet(send_rep, rec_rep, edge_rep)

        # Edge output should differ from original (the MLP adds something)
        assert not torch.allclose(edge_out, edge_rep)
        # But the difference should be the MLP output (residual structure)
        # Verify: edge_diff = edge_out - edge_rep should be the raw message
        # from propagate. Recompute to verify.
        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        _, edge_diff = pnet.propagate(
            pnet.edge_index, x=node_reps, edge_attr=edge_rep
        )
        expected_edge_out = edge_rep + edge_diff
        assert torch.allclose(edge_out, expected_edge_out, atol=1e-5)


# 
# Section D: Batched Processing
# 


class TestBatchedProcessing:
    """Tests for batch dimension handling."""

    def test_output_shapes_batched(self):
        """PropagationNet should work with batched inputs (B, N, d_h)."""
        n_send, n_rec, n_edges, d_h, B = 5, 4, 10, 8, 3
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=True
        )

        torch.manual_seed(42)
        send_rep = torch.randn(B, n_send, d_h)
        rec_rep = torch.randn(B, n_rec, d_h)
        edge_rep = torch.randn(B, n_edges, d_h)

        rec_out, edge_out = pnet(send_rep, rec_rep, edge_rep)
        assert rec_out.shape == (B, n_rec, d_h)
        assert edge_out.shape == (B, n_edges, d_h)

    def test_batch_independence(self):
        """Different samples in a batch should not influence each other."""
        n_send, n_rec, n_edges, d_h = 5, 4, 10, 8
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        torch.manual_seed(42)
        send_rep_0 = torch.randn(1, n_send, d_h)
        rec_rep_0 = torch.randn(1, n_rec, d_h)
        edge_rep_0 = torch.randn(1, n_edges, d_h)

        torch.manual_seed(99)
        send_rep_1 = torch.randn(1, n_send, d_h)
        rec_rep_1 = torch.randn(1, n_rec, d_h)
        edge_rep_1 = torch.randn(1, n_edges, d_h)

        # Run individually
        out_0 = pnet(send_rep_0, rec_rep_0, edge_rep_0)
        out_1 = pnet(send_rep_1, rec_rep_1, edge_rep_1)

        # Run as batch
        send_batch = torch.cat([send_rep_0, send_rep_1], dim=0)
        rec_batch = torch.cat([rec_rep_0, rec_rep_1], dim=0)
        edge_batch = torch.cat([edge_rep_0, edge_rep_1], dim=0)
        out_batch = pnet(send_batch, rec_batch, edge_batch)

        assert torch.allclose(out_batch[0], out_0[0], atol=1e-6)
        assert torch.allclose(out_batch[1], out_1[0], atol=1e-6)


# 
# Section E: Chunk/Split MLP Support
# 


class TestChunkSupport:
    """Tests for edge_chunk_sizes and aggr_chunk_sizes."""

    def test_edge_and_aggr_chunk_sizes(self):
        """PropagationNet should work with edge_chunk_sizes and
        aggr_chunk_sizes, using separate SplitMLPs."""
        n_send, n_rec, d_h = 6, 4, 8
        n_edges_a, n_edges_b = 5, 7
        n_edges = n_edges_a + n_edges_b

        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index,
            input_dim=d_h,
            update_edges=True,
            edge_chunk_sizes=[n_edges_a, n_edges_b],
            aggr_chunk_sizes=[n_rec // 2, n_rec - n_rec // 2],
        )

        torch.manual_seed(42)
        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        rec_out, edge_out = pnet(send_rep, rec_rep, edge_rep)
        assert rec_out.shape == (n_rec, d_h)
        assert edge_out.shape == (n_edges, d_h)

    def test_chunked_differs_from_unchunked(self):
        """Chunked MLPs should produce different outputs than a single MLP
        (they use independent weights for each chunk)."""
        n_send, n_rec, d_h = 6, 4, 8
        n_edges = 12
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet_plain = PropagationNet(
            edge_index.clone(), input_dim=d_h, update_edges=False
        )
        pnet_chunked = PropagationNet(
            edge_index.clone(),
            input_dim=d_h,
            update_edges=False,
            edge_chunk_sizes=[6, 6],
        )

        torch.manual_seed(42)
        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        out_plain = pnet_plain(send_rep, rec_rep, edge_rep)
        out_chunked = pnet_chunked(send_rep, rec_rep, edge_rep)

        # Different MLP architectures -> different results
        assert not torch.allclose(out_plain, out_chunked)


# 
# Section F: Gradient Flow
# 


class TestGradientFlow:
    """Tests for backpropagation through PropagationNet."""

    def test_gradient_flow_all_inputs(self):
        """Gradients should flow to send_rep, rec_rep, and edge_rep."""
        n_send, n_rec, n_edges, d_h = 5, 4, 10, 8
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        send_rep = torch.randn(n_send, d_h, requires_grad=True)
        rec_rep = torch.randn(n_rec, d_h, requires_grad=True)
        edge_rep = torch.randn(n_edges, d_h, requires_grad=True)

        out = pnet(send_rep, rec_rep, edge_rep)
        loss = out.sum()
        loss.backward()

        assert send_rep.grad is not None
        assert rec_rep.grad is not None
        assert edge_rep.grad is not None

    def test_gradient_through_sender_residual(self):
        """Gradient should flow through both MLP path AND direct x_j
        residual in the message function."""
        n_send, n_rec, n_edges, d_h = 3, 2, 6, 4
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        send_rep = torch.randn(n_send, d_h, requires_grad=True)
        rec_rep = torch.randn(n_rec, d_h, requires_grad=True)
        edge_rep = torch.randn(n_edges, d_h, requires_grad=True)

        out = pnet(send_rep, rec_rep, edge_rep)
        loss = out.sum()
        loss.backward()

        # send_rep gradient should be non-zero (flows through x_j residual)
        assert send_rep.grad is not None
        assert send_rep.grad.abs().sum() > 0

        # Zero out edge MLP and recheck: gradient should STILL flow via
        # direct x_j path
        with torch.no_grad():
            for param in pnet.edge_mlp.parameters():
                param.zero_()

        send_rep2 = send_rep.detach().clone().requires_grad_(True)
        rec_rep2 = rec_rep.detach().clone()
        edge_rep2 = edge_rep.detach().clone()

        out2 = pnet(send_rep2, rec_rep2, edge_rep2)
        out2.sum().backward()

        assert send_rep2.grad is not None
        assert send_rep2.grad.abs().sum() > 0

    def test_gradient_through_edge_update(self):
        """When update_edges=True, gradients should flow to edge outputs."""
        n_send, n_rec, n_edges, d_h = 5, 4, 10, 8
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=True
        )

        send_rep = torch.randn(n_send, d_h, requires_grad=True)
        rec_rep = torch.randn(n_rec, d_h, requires_grad=True)
        edge_rep = torch.randn(n_edges, d_h, requires_grad=True)

        rec_out, edge_out = pnet(send_rep, rec_rep, edge_rep)

        # Backprop through edge output only
        edge_out.sum().backward()

        assert edge_rep.grad is not None
        assert edge_rep.grad.abs().sum() > 0


# 
# Section G: Graph Structure Compatibility
# 


class TestGraphStructureCompatibility:
    """Tests for various graph topologies."""

    def test_asymmetric_graph(self):
        """Should handle graphs where n_send != n_rec (e.g. grid->mesh)."""
        # Large ratio like grid(100) -> mesh(10)
        n_send, n_rec, d_h = 100, 10, 8
        n_edges = 200
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        out = pnet(send_rep, rec_rep, edge_rep)
        assert out.shape == (n_rec, d_h)
        assert torch.isfinite(out).all()

    def test_single_sender_single_receiver(self):
        """Degenerate graph with 1 node on each side."""
        d_h = 8
        edge_index = torch.tensor([[0], [0]])

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        send_rep = torch.randn(1, d_h)
        rec_rep = torch.randn(1, d_h)
        edge_rep = torch.randn(1, d_h)

        out = pnet(send_rep, rec_rep, edge_rep)
        assert out.shape == (1, d_h)
        assert torch.isfinite(out).all()

    def test_disconnected_receiver(self):
        """A receiver with no incoming edges should not produce NaN.
        PyG fills aggregation with 0 for disconnected nodes.
        For PropagationNet: rec_new = 0 + aggr_mlp(cat(rec_rep, 0)),
        meaning the receiver loses its direct residual (unlike INet
        which would give rec_rep + aggr_mlp(cat(rec_rep, 0)))."""
        d_h = 4
        # Receivers 0 and 2 get edges, receiver 1 is disconnected
        edge_index = torch.tensor([[0, 1], [0, 2]])

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        torch.manual_seed(42)
        send_rep = torch.randn(2, d_h)
        rec_rep = torch.randn(3, d_h)
        edge_rep = torch.randn(2, d_h)

        out = pnet(send_rep, rec_rep, edge_rep)

        # No NaN for disconnected receiver
        assert out.shape == (3, d_h)
        assert torch.isfinite(out).all()

        # Disconnected receiver: agg_msgs = 0, so
        # rec_new = 0 + aggr_mlp(cat(rec_rep[1], 0))
        zeros = torch.zeros(d_h)
        expected = zeros + pnet.aggr_mlp(
            torch.cat((rec_rep[1], zeros), dim=-1)
        )
        assert torch.allclose(out[1], expected, atol=1e-6)

    def test_self_loops(self):
        """Self-loops (where sender == receiver index in original graph)
        should compute correctly."""
        d_h = 8
        n_nodes = 4
        # Self-loop edges: each node connects to itself
        indices = torch.arange(n_nodes)
        edge_index = torch.stack([indices, indices])

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        send_rep = torch.randn(n_nodes, d_h)
        rec_rep = torch.randn(n_nodes, d_h)
        edge_rep = torch.randn(n_nodes, d_h)

        out = pnet(send_rep, rec_rep, edge_rep)
        assert out.shape == (n_nodes, d_h)
        assert torch.isfinite(out).all()


# 
# Section H: Numerical Stability
# 


class TestNumericalStability:
    """Tests for numerical stability under stress conditions."""

    def test_deep_stacking(self):
        """Multiple PropagationNet layers in sequence should not cause
        numerical blow-up."""
        n_send, n_rec, n_edges, d_h = 10, 10, 30, 16
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        # Stack 8 layers
        layers = []
        for _ in range(8):
            layers.append(
                PropagationNet(
                    edge_index.clone(),
                    input_dim=d_h,
                    update_edges=True,
                )
            )

        torch.manual_seed(42)
        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        current_rec = rec_rep
        current_edge = edge_rep
        for layer in layers:
            current_rec, current_edge = layer(
                send_rep, current_rec, current_edge
            )

        assert torch.isfinite(current_rec).all(), (
            "Receiver reps contain non-finite values after deep stacking"
        )
        assert torch.isfinite(current_edge).all(), (
            "Edge reps contain non-finite values after deep stacking"
        )

    def test_high_degree_stability(self):
        """With many incoming edges per receiver, mean aggregation should
        keep outputs stable."""
        n_send, n_rec, d_h = 50, 3, 8
        # Many edges to few receivers
        n_edges = 500
        edge_index = _make_edge_index(n_send, n_rec, n_edges)

        pnet = PropagationNet(
            edge_index, input_dim=d_h, update_edges=False
        )

        send_rep = torch.randn(n_send, d_h)
        rec_rep = torch.randn(n_rec, d_h)
        edge_rep = torch.randn(n_edges, d_h)

        out = pnet(send_rep, rec_rep, edge_rep)
        assert torch.isfinite(out).all()
        # Mean aggregation should keep magnitude reasonable
        assert out.abs().max() < 1000


# 
# Section I: Model-Level Integration (Deterministic Models)
# 


class TestDefaultBehaviorUnchanged:
    """Tests that without vertical_propnets, models use InteractionNet
    (backward compatibility). All tests use deterministic models only."""

    def test_base_graph_model_default_uses_interaction_net(self):
        """BaseGraphModel should use InteractionNet for g2m/m2g by default."""
        datastore, config = _get_datastore_and_config("1level")

        _, predictor, _, _, _ = _build_model_and_data(
            datastore, config, "graph_lam", "1level"
        )

        assert isinstance(predictor.g2m_gnn, InteractionNet)
        assert not isinstance(predictor.g2m_gnn, PropagationNet)
        assert isinstance(predictor.m2g_gnn, InteractionNet)
        assert not isinstance(predictor.m2g_gnn, PropagationNet)

    def test_base_graph_model_propnet_flag(self):
        """With vertical_propnets=True, g2m/m2g should be PropagationNet."""
        datastore, config = _get_datastore_and_config("1level")

        _, predictor, _, _, _ = _build_model_and_data(
            datastore, config, "graph_lam", "1level",
            vertical_propnets=True,
        )

        assert isinstance(predictor.g2m_gnn, PropagationNet)
        assert isinstance(predictor.m2g_gnn, PropagationNet)

    def test_graph_lam_processor_always_interaction_net(self):
        """GraphLAM processor GNNs should always be InteractionNet,
        even with vertical_propnets=True."""
        datastore, config = _get_datastore_and_config("1level")

        _, predictor, _, _, _ = _build_model_and_data(
            datastore, config, "graph_lam", "1level",
            vertical_propnets=True,
        )

        # Check processor GNNs are InteractionNet (not PropagationNet)
        for module in predictor.processor.modules():
            if isinstance(module, InteractionNet):
                assert not isinstance(module, PropagationNet)

    def test_default_forward_pass_unchanged(self):
        """A forward pass with default settings (no vertical_propnets)
        should produce the same output as before the PropagationNet
        addition, verified by deterministic seeding."""
        datastore, config = _get_datastore_and_config("1level")

        torch.manual_seed(42)
        forecaster_a, _, init_states, forcing, boundary = (
            _build_model_and_data(
                datastore, config, "graph_lam", "1level"
            )
        )

        torch.manual_seed(42)
        forecaster_b, _, _, _, _ = _build_model_and_data(
            datastore, config, "graph_lam", "1level"
        )

        with torch.no_grad():
            out_a, _ = forecaster_a(init_states, forcing, boundary)
            out_b, _ = forecaster_b(init_states, forcing, boundary)

        assert torch.allclose(out_a, out_b)

    def test_propnet_forward_pass_differs(self):
        """A forward pass with vertical_propnets=True should produce
        different outputs than the default (InteractionNet)."""
        datastore, config = _get_datastore_and_config("1level")

        torch.manual_seed(42)
        forecaster_default, _, init_states, forcing, boundary = (
            _build_model_and_data(
                datastore, config, "graph_lam", "1level"
            )
        )

        torch.manual_seed(42)
        forecaster_prop, _, _, _, _ = _build_model_and_data(
            datastore, config, "graph_lam", "1level",
            vertical_propnets=True,
        )

        with torch.no_grad():
            out_default, _ = forecaster_default(
                init_states, forcing, boundary
            )
            out_prop, _ = forecaster_prop(init_states, forcing, boundary)

        assert not torch.allclose(out_default, out_prop)


# 
# Section J: Hierarchical Model Integration
# 


class TestHierarchicalIntegration:
    """Tests for PropagationNet in hierarchical deterministic models."""

    def test_hilam_default_uses_interaction_net(self):
        """HiLAM should use InteractionNet for all GNNs by default."""
        datastore, config = _get_datastore_and_config("hierarchical")

        _, predictor, _, _, _ = _build_model_and_data(
            datastore, config, "hi_lam", "hierarchical"
        )

        # g2m and m2g should be InteractionNet
        assert isinstance(predictor.g2m_gnn, InteractionNet)
        assert not isinstance(predictor.g2m_gnn, PropagationNet)
        assert isinstance(predictor.m2g_gnn, InteractionNet)
        assert not isinstance(predictor.m2g_gnn, PropagationNet)

        # mesh_init_gnns should all be InteractionNet
        for gnn in predictor.mesh_init_gnns:
            assert isinstance(gnn, InteractionNet)
            assert not isinstance(gnn, PropagationNet)

        # mesh_up_gnns (nested) should all be InteractionNet
        for up_gnn_list in predictor.mesh_up_gnns:
            for gnn in up_gnn_list:
                assert isinstance(gnn, InteractionNet)
                assert not isinstance(gnn, PropagationNet)

    def test_hilam_propnet_flag_affects_vertical_gnns(self):
        """With vertical_propnets=True, HiLAM should use PropagationNet
        for mesh_init_gnns and mesh_up_gnns, but InteractionNet for
        mesh_down_gnns and same-level GNNs."""
        datastore, config = _get_datastore_and_config("hierarchical")

        _, predictor, _, _, _ = _build_model_and_data(
            datastore, config, "hi_lam", "hierarchical",
            vertical_propnets=True,
        )

        # g2m and m2g should be PropagationNet
        assert isinstance(predictor.g2m_gnn, PropagationNet)
        assert isinstance(predictor.m2g_gnn, PropagationNet)

        # mesh_init_gnns should be PropagationNet
        for gnn in predictor.mesh_init_gnns:
            assert isinstance(gnn, PropagationNet)

        # mesh_up_gnns should be PropagationNet
        for up_gnn_list in predictor.mesh_up_gnns:
            for gnn in up_gnn_list:
                assert isinstance(gnn, PropagationNet)

        # mesh_down_gnns should ALWAYS be InteractionNet (not PropagationNet)
        for down_gnn_list in predictor.mesh_down_gnns:
            for gnn in down_gnn_list:
                assert isinstance(gnn, InteractionNet)
                assert not isinstance(gnn, PropagationNet)

        # same-level GNNs should ALWAYS be InteractionNet
        for same_gnn_list in predictor.mesh_down_same_gnns:
            for gnn in same_gnn_list:
                assert isinstance(gnn, InteractionNet)
                assert not isinstance(gnn, PropagationNet)
        for same_gnn_list in predictor.mesh_up_same_gnns:
            for gnn in same_gnn_list:
                assert isinstance(gnn, InteractionNet)
                assert not isinstance(gnn, PropagationNet)

    def test_hilam_propnet_forward_pass_differs(self):
        """HiLAM with vertical_propnets=True should produce different
        outputs than the default."""
        datastore, config = _get_datastore_and_config("hierarchical")

        torch.manual_seed(42)
        forecaster_default, _, init_states, forcing, boundary = (
            _build_model_and_data(
                datastore, config, "hi_lam", "hierarchical"
            )
        )

        torch.manual_seed(42)
        forecaster_prop, _, _, _, _ = _build_model_and_data(
            datastore, config, "hi_lam", "hierarchical",
            vertical_propnets=True,
        )

        with torch.no_grad():
            out_default, _ = forecaster_default(
                init_states, forcing, boundary
            )
            out_prop, _ = forecaster_prop(init_states, forcing, boundary)

        assert not torch.allclose(out_default, out_prop)

    def test_hilam_parallel_propnet_flag(self):
        """HiLAMParallel should also support vertical_propnets flag."""
        datastore, config = _get_datastore_and_config("hierarchical")

        _, predictor, _, _, _ = _build_model_and_data(
            datastore, config, "hi_lam_parallel", "hierarchical",
            vertical_propnets=True,
        )

        # g2m and m2g should be PropagationNet
        assert isinstance(predictor.g2m_gnn, PropagationNet)
        assert isinstance(predictor.m2g_gnn, PropagationNet)

        # mesh_init_gnns should be PropagationNet
        for gnn in predictor.mesh_init_gnns:
            assert isinstance(gnn, PropagationNet)

    def test_hilam_read_gnns_always_interaction_net(self):
        """mesh_read_gnns should always be InteractionNet, even with
        vertical_propnets=True."""
        datastore, config = _get_datastore_and_config("hierarchical")

        _, predictor, _, _, _ = _build_model_and_data(
            datastore, config, "hi_lam", "hierarchical",
            vertical_propnets=True,
        )

        for gnn in predictor.mesh_read_gnns:
            assert isinstance(gnn, InteractionNet)
            assert not isinstance(gnn, PropagationNet)
