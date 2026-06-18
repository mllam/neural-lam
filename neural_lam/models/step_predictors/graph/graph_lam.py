"""Graph-based LAM model with a flat mesh."""

# Standard library

# Third-party
import torch
import torch_geometric as pyg

# Local
from .... import utils
from ....datastore import BaseDatastore
from ....gnn_layers import InteractionNet
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
        output_clamping_lower: dict[str, float] | None = None,
        output_clamping_upper: dict[str, float] | None = None,
        g2m_gnn_type: str = "InteractionNet",
        m2g_gnn_type: str = "InteractionNet",
    ) -> None:
        """
        Initialize the GraphLAM model.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore providing grid metadata and data access.
        graph_name : str, default "multiscale"
            The name of the graph to load.
        hidden_dim : int, default 64
            The dimension of the hidden representations.
        hidden_layers : int, default 1
            The number of hidden layers in the MLPs.
        processor_layers : int, default 4
            The number of processor layers in the GNN.
        mesh_aggr : str, default "sum"
            The aggregation method for mesh nodes.
        num_past_forcing_steps : int, default 1
            The number of past forcing steps to include.
        num_future_forcing_steps : int, default 1
            The number of future forcing steps to include.
        output_std : bool, default False
            Whether to output a predicted standard deviation.
        output_clamping_lower : dict, optional
            Lower clamping limits for state variables.
        output_clamping_upper : dict, optional
            Upper clamping limits for state variables.
        """
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
            g2m_gnn_type=g2m_gnn_type,
            m2g_gnn_type=m2g_gnn_type,
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
                (
                    net,
                    "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep",
                )
                for net in processor_nets
            ],
        )

    def get_num_mesh(self) -> tuple[int, int]:
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding.

        Returns
        -------
        num_mesh_nodes : int
            The number of mesh nodes.
        num_ignore_mesh_nodes : int
            The number of mesh nodes to ignore.
        """
        return self.mesh_static_features.shape[0], 0

    def embedd_mesh_nodes(self) -> torch.Tensor:
        """
        Embed static mesh node features.

        Returns
        -------
        torch.Tensor
            Shape ``(num_mesh_nodes, hidden_dim)``. Embedded mesh node
            representations. Dims: ``num_mesh_nodes`` is the number of
            mesh nodes and ``hidden_dim`` is the hidden dimension.
        """
        return self.mesh_embedder(
            self.mesh_static_features
        )  # (num_mesh_nodes, hidden_dim)

    def process_step(self, mesh_rep: torch.Tensor) -> torch.Tensor:
        """
        Process the mesh representation through the flat message-passing
        processor (all nodes at a single level).

        Parameters
        ----------
        mesh_rep : torch.Tensor
            Shape ``(B, num_mesh_nodes, hidden_dim)``. Current mesh node
            representations. Dims: ``B`` is batch size, ``num_mesh_nodes`` is
            the number of mesh nodes, and ``hidden_dim`` is the hidden
            dimension.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_mesh_nodes, hidden_dim)``. Updated mesh node
            representations. Dims: same as ``mesh_rep``.
        """
        # Embed m2m here first
        batch_size = mesh_rep.shape[0]
        m2m_emb = self.m2m_embedder(
            self.m2m_features
        )  # (num_edges, hidden_dim)
        m2m_emb_expanded = self.expand_to_batch(
            m2m_emb, batch_size
        )  # (B, num_edges, hidden_dim)

        mesh_rep, _ = self.processor(
            mesh_rep, m2m_emb_expanded
        )  # (B, num_mesh_nodes, hidden_dim)
        return mesh_rep
