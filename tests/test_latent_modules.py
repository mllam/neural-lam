"""Unit tests for the latent encoder/decoder infrastructure.

These tests exercise the latent modules in isolation with synthetic edge
indices and tensor inputs, so they do not depend on any datastore or graph
fixture. They cover output shapes, distribution properties and that
backpropagation reaches all parameters.
"""

# Third-party
import pytest
import torch

# First-party
from neural_lam.models.latent import (
    BaseLatentEncoder,
    ConstantLatentEncoder,
    GraphLatentDecoder,
    GraphLatentEncoder,
    HiGraphLatentDecoder,
    HiGraphLatentEncoder,
)
from neural_lam.utils import IdentityModule, make_gnn_seq


def _fully_connected_edge_index(n_send, n_rec):
    senders = (
        torch.arange(n_send).unsqueeze(1).expand(n_send, n_rec).reshape(-1)
    )
    receivers = (
        torch.arange(n_rec).unsqueeze(0).expand(n_send, n_rec).reshape(-1)
    )
    return torch.stack([senders, receivers])


def _assert_every_param_has_grad(module):
    """Fail if any trainable parameter on ``module`` received no gradient.

    Catches dead-param regressions if a future change wires in a sub-module
    that the forward pass never reaches.
    """
    for name, p in module.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"parameter {name} received no gradient"


@pytest.fixture
def flat_dims():
    return {
        "batch_size": 2,
        "num_grid": 5,
        "num_mesh": 3,
        "hidden_dim": 8,
        "latent_dim": 4,
        "num_state_vars": 2,
        "hidden_layers": 1,
        "m2m_layers": 2,
    }


@pytest.fixture
def flat_edges(flat_dims):
    n_grid = flat_dims["num_grid"]
    n_mesh = flat_dims["num_mesh"]
    return {
        "g2m": _fully_connected_edge_index(n_grid, n_mesh),
        "m2m": _fully_connected_edge_index(n_mesh, n_mesh),
        "m2g": _fully_connected_edge_index(n_mesh, n_grid),
    }


@pytest.fixture
def flat_graph_emb(flat_dims, flat_edges):
    B = flat_dims["batch_size"]
    d_h = flat_dims["hidden_dim"]
    return {
        "mesh": torch.randn(B, flat_dims["num_mesh"], d_h),
        "g2m": torch.randn(B, flat_edges["g2m"].shape[1], d_h),
        "m2m": torch.randn(B, flat_edges["m2m"].shape[1], d_h),
        "m2g": torch.randn(B, flat_edges["m2g"].shape[1], d_h),
    }


def test_identity_module_passes_args_through():
    module = IdentityModule()
    a, b, c = torch.randn(3), torch.randn(2), torch.randn(1)
    out = module(a, b, c)
    assert out == (a, b, c)


def test_make_gnn_seq_zero_layers_raises():
    """make_gnn_seq must build a real sequence; the no-op (identity) case is
    the caller's responsibility, exercised via the zero-intra-layer tests."""
    edge_index = _fully_connected_edge_index(3, 3)
    with pytest.raises(ValueError, match="num_gnn_layers >= 1"):
        make_gnn_seq(
            edge_index, num_gnn_layers=0, hidden_layers=1, hidden_dim=8
        )


def test_make_gnn_seq_positive_layers_runs():
    edge_index = _fully_connected_edge_index(3, 3)
    seq = make_gnn_seq(
        edge_index, num_gnn_layers=2, hidden_layers=1, hidden_dim=8
    )
    mesh_rep = torch.randn(2, 3, 8)
    edge_rep = torch.randn(2, edge_index.shape[1], 8)
    out_mesh, out_edge = seq(mesh_rep, edge_rep)
    assert out_mesh.shape == mesh_rep.shape
    assert out_edge.shape == edge_rep.shape


class _IdentityEncoder(BaseLatentEncoder):
    """Trivial encoder used to verify BaseLatentEncoder distribution logic."""

    def __init__(self, latent_dim, num_mesh_nodes, output_dist):
        super().__init__(latent_dim, output_dist)
        self.num_mesh_nodes = num_mesh_nodes
        # Learnable params so we can verify backprop reaches them
        self.bias = torch.nn.Parameter(torch.zeros(self.output_dim))

    def compute_dist_params(self, grid_rep, **kwargs):
        B = grid_rep.shape[0]
        return self.bias.expand(B, self.num_mesh_nodes, self.output_dim)


def test_base_encoder_isotropic_has_unit_std():
    enc = _IdentityEncoder(
        latent_dim=4, num_mesh_nodes=3, output_dist="isotropic"
    )
    grid_rep = torch.randn(2, 5, 8)
    dist = enc(grid_rep)
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.mean.shape == (2, 3, 4)
    assert torch.allclose(dist.stddev, torch.ones_like(dist.stddev))


def test_base_encoder_diagonal_has_positive_std():
    enc = _IdentityEncoder(
        latent_dim=4, num_mesh_nodes=3, output_dist="diagonal"
    )
    grid_rep = torch.randn(2, 5, 8)
    dist = enc(grid_rep)
    assert dist.mean.shape == (2, 3, 4)
    assert dist.stddev.shape == (2, 3, 4)
    # softplus(0) + eps must be strictly positive
    assert (dist.stddev > 0).all()


def test_base_encoder_rejects_unknown_dist():
    with pytest.raises(ValueError):
        _IdentityEncoder(latent_dim=4, num_mesh_nodes=3, output_dist="bogus")


def test_constant_encoder_is_input_independent():
    enc = ConstantLatentEncoder(
        latent_dim=4, num_mesh_nodes=3, output_dist="isotropic"
    )
    a = enc(torch.randn(2, 5, 8))
    b = enc(torch.randn(2, 5, 8) * 100)
    assert torch.equal(a.mean, b.mean)
    assert torch.equal(a.stddev, b.stddev)
    assert a.mean.shape == (2, 3, 4)
    # Prior is a mean-0 standard normal (isotropic): fixes the prob_model_lam
    # mean-1 bug, see ConstantLatentEncoder docstring.
    assert torch.equal(a.mean, torch.zeros_like(a.mean))
    assert torch.allclose(a.stddev, torch.ones_like(a.stddev))


def test_graph_encoder_shapes_and_backprop(
    flat_dims, flat_edges, flat_graph_emb
):
    enc = GraphLatentEncoder(
        latent_dim=flat_dims["latent_dim"],
        g2m_edge_index=flat_edges["g2m"],
        m2m_edge_index=flat_edges["m2m"],
        hidden_dim=flat_dims["hidden_dim"],
        m2m_layers=flat_dims["m2m_layers"],
        hidden_layers=flat_dims["hidden_layers"],
        output_dist="diagonal",
    )
    grid_rep = torch.randn(
        flat_dims["batch_size"],
        flat_dims["num_grid"],
        flat_dims["hidden_dim"],
    )
    dist = enc(grid_rep, graph_emb=flat_graph_emb)
    assert dist.mean.shape == (
        flat_dims["batch_size"],
        flat_dims["num_mesh"],
        flat_dims["latent_dim"],
    )

    dist.rsample().sum().backward()
    _assert_every_param_has_grad(enc)


def test_graph_decoder_shapes_with_output_std(
    flat_dims, flat_edges, flat_graph_emb
):
    dec = GraphLatentDecoder(
        g2m_edge_index=flat_edges["g2m"],
        m2m_edge_index=flat_edges["m2m"],
        m2g_edge_index=flat_edges["m2g"],
        hidden_dim=flat_dims["hidden_dim"],
        latent_dim=flat_dims["latent_dim"],
        num_state_vars=flat_dims["num_state_vars"],
        m2m_layers=flat_dims["m2m_layers"],
        hidden_layers=flat_dims["hidden_layers"],
        output_std=True,
    )
    B = flat_dims["batch_size"]
    grid_rep = torch.randn(B, flat_dims["num_grid"], flat_dims["hidden_dim"])
    latent_samples = torch.randn(
        B, flat_dims["num_mesh"], flat_dims["latent_dim"]
    )
    last_state = torch.randn(
        B, flat_dims["num_grid"], flat_dims["num_state_vars"]
    )

    pred_mean, pred_std = dec(
        grid_rep, latent_samples, last_state, flat_graph_emb
    )

    expected_shape = (B, flat_dims["num_grid"], flat_dims["num_state_vars"])
    assert pred_mean.shape == expected_shape
    assert pred_std is not None
    assert pred_std.shape == expected_shape
    assert (pred_std > 0).all()

    (pred_mean.sum() + pred_std.sum()).backward()
    _assert_every_param_has_grad(dec)


def test_graph_decoder_no_output_std_returns_none(
    flat_dims, flat_edges, flat_graph_emb
):
    dec = GraphLatentDecoder(
        g2m_edge_index=flat_edges["g2m"],
        m2m_edge_index=flat_edges["m2m"],
        m2g_edge_index=flat_edges["m2g"],
        hidden_dim=flat_dims["hidden_dim"],
        latent_dim=flat_dims["latent_dim"],
        num_state_vars=flat_dims["num_state_vars"],
        m2m_layers=flat_dims["m2m_layers"],
        hidden_layers=flat_dims["hidden_layers"],
        output_std=False,
    )
    B = flat_dims["batch_size"]
    grid_rep = torch.randn(B, flat_dims["num_grid"], flat_dims["hidden_dim"])
    latent_samples = torch.randn(
        B, flat_dims["num_mesh"], flat_dims["latent_dim"]
    )
    last_state = torch.randn(
        B, flat_dims["num_grid"], flat_dims["num_state_vars"]
    )

    pred_mean, pred_std = dec(
        grid_rep, latent_samples, last_state, flat_graph_emb
    )
    assert pred_mean.shape == (
        B,
        flat_dims["num_grid"],
        flat_dims["num_state_vars"],
    )
    assert pred_std is None


def test_flat_modules_zero_m2m_layers_use_identity(
    flat_dims, flat_edges, flat_graph_emb
):
    """m2m_layers=0 routes on-mesh processing through IdentityModule at the
    call site (make_gnn_seq itself rejects 0). Exercise both flat modules."""
    enc = GraphLatentEncoder(
        latent_dim=flat_dims["latent_dim"],
        g2m_edge_index=flat_edges["g2m"],
        m2m_edge_index=flat_edges["m2m"],
        hidden_dim=flat_dims["hidden_dim"],
        m2m_layers=0,
        hidden_layers=flat_dims["hidden_layers"],
    )
    assert isinstance(enc.m2m_gnns, IdentityModule)

    dec = GraphLatentDecoder(
        g2m_edge_index=flat_edges["g2m"],
        m2m_edge_index=flat_edges["m2m"],
        m2g_edge_index=flat_edges["m2g"],
        hidden_dim=flat_dims["hidden_dim"],
        latent_dim=flat_dims["latent_dim"],
        num_state_vars=flat_dims["num_state_vars"],
        m2m_layers=0,
        hidden_layers=flat_dims["hidden_layers"],
    )
    assert isinstance(dec.m2m_gnns, IdentityModule)

    B = flat_dims["batch_size"]
    grid_rep = torch.randn(B, flat_dims["num_grid"], flat_dims["hidden_dim"])
    dist = enc(grid_rep, graph_emb=flat_graph_emb)
    assert dist.mean.shape == (
        B,
        flat_dims["num_mesh"],
        flat_dims["latent_dim"],
    )

    latent_samples = torch.randn(
        B, flat_dims["num_mesh"], flat_dims["latent_dim"]
    )
    last_state = torch.randn(
        B, flat_dims["num_grid"], flat_dims["num_state_vars"]
    )
    pred_mean, _ = dec(grid_rep, latent_samples, last_state, flat_graph_emb)
    assert pred_mean.shape == (
        B,
        flat_dims["num_grid"],
        flat_dims["num_state_vars"],
    )


# --- Hierarchical fixtures and tests ----------------------------------------


@pytest.fixture
def hi_dims():
    return {
        "batch_size": 2,
        "num_grid": 5,
        "mesh_per_level": [4, 3],  # bottom -> top
        "hidden_dim": 8,
        "latent_dim": 4,
        "num_state_vars": 2,
        "hidden_layers": 1,
        "intra_level_layers": 1,
    }


@pytest.fixture
def hi_edges(hi_dims):
    bot, top = hi_dims["mesh_per_level"]
    n_grid = hi_dims["num_grid"]
    return {
        "g2m": _fully_connected_edge_index(n_grid, bot),
        "m2g": _fully_connected_edge_index(bot, n_grid),
        "m2m": [
            _fully_connected_edge_index(bot, bot),
            _fully_connected_edge_index(top, top),
        ],
        "mesh_up": [_fully_connected_edge_index(bot, top)],
        "mesh_down": [_fully_connected_edge_index(top, bot)],
    }


@pytest.fixture
def hi_graph_emb(hi_dims, hi_edges):
    B = hi_dims["batch_size"]
    d_h = hi_dims["hidden_dim"]
    return {
        "mesh": [torch.randn(B, n, d_h) for n in hi_dims["mesh_per_level"]],
        "g2m": torch.randn(B, hi_edges["g2m"].shape[1], d_h),
        "m2g": torch.randn(B, hi_edges["m2g"].shape[1], d_h),
        "m2m": [torch.randn(B, e.shape[1], d_h) for e in hi_edges["m2m"]],
        "mesh_up": [
            torch.randn(B, e.shape[1], d_h) for e in hi_edges["mesh_up"]
        ],
        "mesh_down": [
            torch.randn(B, e.shape[1], d_h) for e in hi_edges["mesh_down"]
        ],
    }


def test_hi_graph_encoder_shape_at_top_level(hi_dims, hi_edges, hi_graph_emb):
    enc = HiGraphLatentEncoder(
        latent_dim=hi_dims["latent_dim"],
        g2m_edge_index=hi_edges["g2m"],
        m2m_edge_index=hi_edges["m2m"],
        mesh_up_edge_index=hi_edges["mesh_up"],
        hidden_dim=hi_dims["hidden_dim"],
        intra_level_layers=hi_dims["intra_level_layers"],
        hidden_layers=hi_dims["hidden_layers"],
        output_dist="diagonal",
    )
    grid_rep = torch.randn(
        hi_dims["batch_size"], hi_dims["num_grid"], hi_dims["hidden_dim"]
    )
    dist = enc(grid_rep, graph_emb=hi_graph_emb)
    top_n = hi_dims["mesh_per_level"][-1]
    assert dist.mean.shape == (
        hi_dims["batch_size"],
        top_n,
        hi_dims["latent_dim"],
    )
    assert (dist.stddev > 0).all()


def test_hi_graph_decoder_shape_back_to_grid(hi_dims, hi_edges, hi_graph_emb):
    dec = HiGraphLatentDecoder(
        g2m_edge_index=hi_edges["g2m"],
        m2m_edge_index=hi_edges["m2m"],
        m2g_edge_index=hi_edges["m2g"],
        mesh_up_edge_index=hi_edges["mesh_up"],
        mesh_down_edge_index=hi_edges["mesh_down"],
        hidden_dim=hi_dims["hidden_dim"],
        latent_dim=hi_dims["latent_dim"],
        num_state_vars=hi_dims["num_state_vars"],
        intra_level_layers=hi_dims["intra_level_layers"],
        hidden_layers=hi_dims["hidden_layers"],
        output_std=True,
    )
    B = hi_dims["batch_size"]
    top_n = hi_dims["mesh_per_level"][-1]
    grid_rep = torch.randn(B, hi_dims["num_grid"], hi_dims["hidden_dim"])
    latent_samples = torch.randn(B, top_n, hi_dims["latent_dim"])
    last_state = torch.randn(B, hi_dims["num_grid"], hi_dims["num_state_vars"])

    pred_mean, pred_std = dec(
        grid_rep, latent_samples, last_state, hi_graph_emb
    )
    expected_shape = (B, hi_dims["num_grid"], hi_dims["num_state_vars"])
    assert pred_mean.shape == expected_shape
    assert pred_std.shape == expected_shape
    assert (pred_std > 0).all()


def _build_hi_inputs(mesh_per_level, num_grid, hidden_dim, batch_size):
    """Construct edge indices and graph_emb for an arbitrary mesh hierarchy."""
    bot = mesh_per_level[0]
    edges = {
        "g2m": _fully_connected_edge_index(num_grid, bot),
        "m2g": _fully_connected_edge_index(bot, num_grid),
        "m2m": [_fully_connected_edge_index(n, n) for n in mesh_per_level],
        "mesh_up": [
            _fully_connected_edge_index(lo, hi)
            for lo, hi in zip(mesh_per_level[:-1], mesh_per_level[1:])
        ],
        "mesh_down": [
            _fully_connected_edge_index(hi, lo)
            for lo, hi in zip(mesh_per_level[:-1], mesh_per_level[1:])
        ],
    }
    B, d_h = batch_size, hidden_dim
    graph_emb = {
        "mesh": [torch.randn(B, n, d_h) for n in mesh_per_level],
        "g2m": torch.randn(B, edges["g2m"].shape[1], d_h),
        "m2g": torch.randn(B, edges["m2g"].shape[1], d_h),
        "m2m": [torch.randn(B, e.shape[1], d_h) for e in edges["m2m"]],
        "mesh_up": [torch.randn(B, e.shape[1], d_h) for e in edges["mesh_up"]],
        "mesh_down": [
            torch.randn(B, e.shape[1], d_h) for e in edges["mesh_down"]
        ],
    }
    return edges, graph_emb


def test_hi_graph_decoder_three_levels():
    """Three-level hierarchy exercises non-empty intra_down loop and the full
    up/down recursion, which num_levels=2 only partially covers."""
    mesh_per_level = [5, 4, 3]
    B, d_h, latent_dim, num_state_vars = 2, 8, 4, 2
    num_grid = 6
    edges, graph_emb = _build_hi_inputs(mesh_per_level, num_grid, d_h, B)

    dec = HiGraphLatentDecoder(
        g2m_edge_index=edges["g2m"],
        m2m_edge_index=edges["m2m"],
        m2g_edge_index=edges["m2g"],
        mesh_up_edge_index=edges["mesh_up"],
        mesh_down_edge_index=edges["mesh_down"],
        hidden_dim=d_h,
        latent_dim=latent_dim,
        num_state_vars=num_state_vars,
        intra_level_layers=1,
        hidden_layers=1,
        output_std=True,
    )
    grid_rep = torch.randn(B, num_grid, d_h)
    top_n = mesh_per_level[-1]
    latent_samples = torch.randn(B, top_n, latent_dim)
    last_state = torch.randn(B, num_grid, num_state_vars)

    pred_mean, pred_std = dec(grid_rep, latent_samples, last_state, graph_emb)
    assert pred_mean.shape == (B, num_grid, num_state_vars)
    assert pred_std.shape == (B, num_grid, num_state_vars)

    (pred_mean.sum() + pred_std.sum()).backward()
    _assert_every_param_has_grad(dec)


def test_hi_graph_encoder_three_levels():
    mesh_per_level = [5, 4, 3]
    B, d_h, latent_dim = 2, 8, 4
    num_grid = 6
    edges, graph_emb = _build_hi_inputs(mesh_per_level, num_grid, d_h, B)

    enc = HiGraphLatentEncoder(
        latent_dim=latent_dim,
        g2m_edge_index=edges["g2m"],
        m2m_edge_index=edges["m2m"],
        mesh_up_edge_index=edges["mesh_up"],
        hidden_dim=d_h,
        intra_level_layers=1,
        hidden_layers=1,
        output_dist="diagonal",
    )
    grid_rep = torch.randn(B, num_grid, d_h)
    dist = enc(grid_rep, graph_emb=graph_emb)
    assert dist.mean.shape == (B, mesh_per_level[-1], latent_dim)

    dist.rsample().sum().backward()
    _assert_every_param_has_grad(enc)


def test_hi_graph_modules_reject_single_level():
    """Hierarchical encoder/decoder must refuse a single-level mesh,
    otherwise the latent would be silently ignored."""
    # Single-level mesh: m2m has length 1, mesh_up/mesh_down are empty.
    edges, _ = _build_hi_inputs(
        mesh_per_level=[4], num_grid=5, hidden_dim=8, batch_size=2
    )
    with pytest.raises(ValueError, match="at least 2 mesh levels"):
        HiGraphLatentEncoder(
            latent_dim=4,
            g2m_edge_index=edges["g2m"],
            m2m_edge_index=edges["m2m"],
            mesh_up_edge_index=edges["mesh_up"],
            hidden_dim=8,
            intra_level_layers=1,
        )
    with pytest.raises(ValueError, match="at least 2 mesh levels"):
        HiGraphLatentDecoder(
            g2m_edge_index=edges["g2m"],
            m2m_edge_index=edges["m2m"],
            m2g_edge_index=edges["m2g"],
            mesh_up_edge_index=edges["mesh_up"],
            mesh_down_edge_index=edges["mesh_down"],
            hidden_dim=8,
            latent_dim=4,
            num_state_vars=2,
            intra_level_layers=1,
        )


def test_hi_graph_decoder_zero_intra_layers(hi_dims, hi_edges, hi_graph_emb):
    """intra_level_layers=0 routes intra-processing through IdentityModule
    (the make_gnn_seq branch). Exercise that path end-to-end."""
    dec = HiGraphLatentDecoder(
        g2m_edge_index=hi_edges["g2m"],
        m2m_edge_index=hi_edges["m2m"],
        m2g_edge_index=hi_edges["m2g"],
        mesh_up_edge_index=hi_edges["mesh_up"],
        mesh_down_edge_index=hi_edges["mesh_down"],
        hidden_dim=hi_dims["hidden_dim"],
        latent_dim=hi_dims["latent_dim"],
        num_state_vars=hi_dims["num_state_vars"],
        intra_level_layers=0,
        hidden_layers=hi_dims["hidden_layers"],
        output_std=True,
    )
    B = hi_dims["batch_size"]
    top_n = hi_dims["mesh_per_level"][-1]
    grid_rep = torch.randn(B, hi_dims["num_grid"], hi_dims["hidden_dim"])
    latent_samples = torch.randn(B, top_n, hi_dims["latent_dim"])
    last_state = torch.randn(B, hi_dims["num_grid"], hi_dims["num_state_vars"])

    pred_mean, pred_std = dec(
        grid_rep, latent_samples, last_state, hi_graph_emb
    )
    expected_shape = (B, hi_dims["num_grid"], hi_dims["num_state_vars"])
    assert pred_mean.shape == expected_shape
    assert pred_std.shape == expected_shape


def test_hi_graph_encoder_zero_intra_layers(hi_dims, hi_edges, hi_graph_emb):
    enc = HiGraphLatentEncoder(
        latent_dim=hi_dims["latent_dim"],
        g2m_edge_index=hi_edges["g2m"],
        m2m_edge_index=hi_edges["m2m"],
        mesh_up_edge_index=hi_edges["mesh_up"],
        hidden_dim=hi_dims["hidden_dim"],
        intra_level_layers=0,
        hidden_layers=hi_dims["hidden_layers"],
        output_dist="isotropic",
    )
    grid_rep = torch.randn(
        hi_dims["batch_size"], hi_dims["num_grid"], hi_dims["hidden_dim"]
    )
    dist = enc(grid_rep, graph_emb=hi_graph_emb)
    assert dist.mean.shape == (
        hi_dims["batch_size"],
        hi_dims["mesh_per_level"][-1],
        hi_dims["latent_dim"],
    )
