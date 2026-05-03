# Standard library
from typing import Dict, Optional

# Third-party
import torch_geometric as pyg

# Local
from .... import utils
from ....datastore import BaseDatastore
from ....interaction_net import InteractionNet
from .base import BaseGraphModel


class GraphLAM(BaseGraphModel):
    """
    Full graph-based LAM model that can be used with different
    (non-hierarchical )graphs. Mainly based on GraphCast, but the model from
    Keisler (2022) is almost identical. Used for GC-LAM and L1-LAM in
    Oskarsson et al. (2023).
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        graph_name: str = "multiscale",
        hidden_dim: int = 64,
        hidden_layers: int = 1,
        processor_layers: int = 4,
        mesh_aggr: str = "sum",
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        output_std: bool = False,
        output_clamping_lower: Optional[Dict[str, float]] = None,
        output_clamping_upper: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            datastore=datastore,
            graph_name=graph_name,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            processor_layers=processor_layers,
            mesh_aggr=mesh_aggr,
            num_past_forcing_steps=num_past_forcing_steps,
            num_future_forcing_steps=num_future_forcing_steps,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )

        assert (
            not self.hierarchical
        ), "GraphLAM does not use a hierarchical mesh graph"

        # grid_dim from data + static + batch_static
        mesh_dim = self.mesh_static_features.shape[1]
        m2m_edges, m2m_dim = self.m2m_features.shape
        utils.log_on_rank_zero(
            f"Edges in subgraphs: m2m={m2m_edges}, g2m={self.g2m_edges}, "
            f"m2g={self.m2g_edges}"
        )

        # Define sub-models
        # Feature embedders for mesh
        self.mesh_embedder = utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
        self.m2m_embedder = utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)

        # GNNs
        # processor
        processor_nets = [
            InteractionNet(
                self.m2m_edge_index,
                hidden_dim,
                hidden_layers=hidden_layers,
                aggr=mesh_aggr,
            )
            for _ in range(processor_layers)
        ]
        self.processor = pyg.nn.Sequential(
            "mesh_rep, edge_rep",
            [
                (net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
                for net in processor_nets
            ],
        )

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return self.mesh_static_features.shape[0], 0

    def embedd_mesh_nodes(self):
        """
        Embed static mesh node features.

        Returns
        -------
        torch.Tensor
            Shape ``(N_mesh, d_h)``. Embedded mesh node representations.
            Dims: ``N_mesh`` is the number of mesh nodes and ``d_h`` is
            the hidden dimension.
        """
        return self.mesh_embedder(self.mesh_static_features)  # (N_mesh, d_h)

    def process_step(self, mesh_rep):
        """
        Process the mesh representation through the flat message-passing
        processor (all nodes at a single level).

        Parameters
        ----------
        mesh_rep : torch.Tensor
            Shape ``(B, N_mesh, d_h)``. Current mesh node
            representations. Dims: ``B`` is batch size, ``N_mesh`` is
            the number of mesh nodes, and ``d_h`` is the hidden
            dimension.

        Returns
        -------
        torch.Tensor
            Shape ``(B, N_mesh, d_h)``. Updated mesh node
            representations. Dims: same as ``mesh_rep``.
        """
        # Embed m2m here first
        batch_size = mesh_rep.shape[0]
        m2m_emb = self.m2m_embedder(self.m2m_features)  # (M_mesh, d_h)
        m2m_emb_expanded = self.expand_to_batch(
            m2m_emb, batch_size
        )  # (B, M_mesh, d_h)

        mesh_rep, _ = self.processor(
            mesh_rep, m2m_emb_expanded
        )  # (B, N_mesh, d_h)
        return mesh_rep
