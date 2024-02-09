import torch
import torch_geometric as pyg

from torch.nn import ReLU
from torch_geometric.nn import GCN

from neural_lam import utils
from neural_lam.models.base_graph_model import BaseGraphModel

class GCNModel(BaseGraphModel):
    """
    GCN model for weather forecasting. 
    
    Don't use any edge features for now.
    
    Based on GraphCast, but using GCN instead of InteractionNet for GNN layers.
    """
    def __init__(self, args):
        super().__init__(args)

        assert not self.hierarchical, "GCN Model does not use a hierarchical mesh graph"

        # grid_dim from data + static + batch_static
        mesh_dim = self.mesh_static_features.shape[1]
        m2m_edges, m2m_dim = self.m2m_features.shape
        print(f"Edges in subgraphs: m2m={m2m_edges}, g2m={self.g2m_edges}, "
                f"m2g={self.m2g_edges}")

        # Define sub-models
        # Feature embedders for mesh
        self.mesh_embedder = utils.make_mlp([mesh_dim] +
                self.mlp_blueprint_end)

        # GNNs
        # processor
        self.processor = GCN(
            in_channels=args.hidden_dim,
            hidden_channels=args.hidden_dim,
            num_layers=args.processor_layers,
            out_channels=args.hidden_dim,
            act=ReLU(inplace=True),
        )

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return self.mesh_static_features.shape[0], 0

    def embedd_mesh_nodes(self):
        """
        Embedd static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        return self.mesh_embedder(self.mesh_static_features) # (N_mesh, d_h)

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        # Embedd m2m here first
        # batch_size = mesh_rep.shape[0]
        # m2m_emb = self.m2m_embedder(self.m2m_features) # (M_mesh, d_h)
        # m2m_emb_expanded = self.expand_to_batch(m2m_emb, batch_size) # (B, M_mesh, d_h)

        # mesh_rep, _ = self.processor(mesh_rep, m2m_emb) # (B, N_mesh, d_h)
        mesh_rep = self.processor(mesh_rep, self.m2m_edge_index) # (B, N_mesh, d_h)
        return mesh_rep