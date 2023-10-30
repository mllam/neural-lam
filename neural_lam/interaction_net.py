import torch
import torch_geometric as pyg


class InteractionNet(pyg.nn.MessagePassing):
    """
    Implementation of the GraphCast version of an Interaction Network
    """

    def __init__(self, edge_index, edge_mlp, aggr_mlp, aggr="sum"):
        assert aggr in ("sum", "mean"), f"Unknown aggregation method: {aggr}"
        super().__init__(aggr=aggr)

        self.register_buffer("edge_index", edge_index, persistent=False)
        self.edge_mlp = edge_mlp
        self.aggr_mlp = aggr_mlp

    def forward(self, x, edge_attr):
        self.edge_index = self.edge_index.to(x.device)

        edge_rep_aggr, edge_rep = self.propagate(self.edge_index, x=x,
                                                 edge_attr=edge_attr)
        node_rep = self.aggr_mlp(torch.cat((x, edge_rep_aggr), dim=-1))

        # Residual connections
        node_rep = node_rep + x
        edge_rep = edge_rep + edge_attr

        return node_rep, edge_rep

    def message(self, x_j, x_i, edge_attr):
        """
        Message fromm node j to node i.
        """
        return self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1))

    def aggregate(self, messages, index, ptr, dim_size):
        # Change to return both aggregated and original messages
        aggr = super().aggregate(messages, index, ptr, dim_size)
        return aggr, messages

    def update(self, inputs):
        # Pass two argument input (from aggregate) through this
        return inputs


class EncoderInteractionNet(InteractionNet):
    """
    InteractionNet tailored for grid->mesh encoding step
    """

    def __init__(self, edge_index, edge_mlp, aggr_mlp, grid_mlp, N_mesh, N_mesh_ignore):
        super().__init__(edge_index, edge_mlp, aggr_mlp)
        self.grid_mlp = grid_mlp
        self.N_mesh = N_mesh
        # Number of mesh nodes to not use (e.g. higher level in hierarchy)
        self.N_mesh_ignore = N_mesh_ignore
        self.N_mesh_encode = self.N_mesh - self.N_mesh_ignore

    def aggregate(self, messages, index, ptr, dim_size):
        # Force to only aggregate to mesh nodes
        return super().aggregate(messages, index, ptr, self.N_mesh_encode)

    def forward(self, x, edge_attr):
        self.edge_index = self.edge_index.to(x.device)
        mesh_edge_rep_aggr, _ = self.propagate(self.edge_index, x=x,
                                               edge_attr=edge_attr)

        # Use aggr_mlp only for mesh nodes, grid nodes have no aggregated messages input
        mesh_x = x[:, :self.N_mesh_encode]  # (B, N_mesh_enc, d_h)
        mesh_ignored_x = x[:, self.N_mesh_encode:self.N_mesh]  # (B, N_mesh_ign, d_h)
        grid_x = x[:, self.N_mesh:]  # (B, N_grid, d_h)
        mesh_enc_rep = self.aggr_mlp(torch.cat((mesh_x, mesh_edge_rep_aggr),
                                               dim=-1))  # (B, N_mesh_enc, d_h)

        # Separate MLP for mesh nodes
        grid_rep = self.grid_mlp(grid_x)

        # Residual connections
        mesh_enc_rep = mesh_enc_rep + mesh_x
        grid_rep = grid_rep + grid_x
        # Don't care about edge representation any more

        # Concatenate on the ignored mesh representations
        mesh_rep = torch.cat((mesh_enc_rep, mesh_ignored_x), dim=1)

        return grid_rep, mesh_rep


class DecoderInteractionNet(InteractionNet):
    """
    InteractionNet tailored for mesh->grid decoding step
    """

    def __init__(self, edge_index, edge_mlp, aggr_mlp, N_mesh, N_grid):
        super().__init__(edge_index, edge_mlp, aggr_mlp)
        self.N_mesh = N_mesh
        self.N_grid = N_grid

    def aggregate(self, messages, index, ptr, dim_size):
        shifted_index = index - self.N_mesh  # Shift N_mesh->0 so we start aggr. to grid
        return super().aggregate(messages, shifted_index, ptr, self.N_grid)

    def forward(self, x, edge_attr):
        self.edge_index = self.edge_index.to(x.device)

        grid_edge_rep_aggr, _ = self.propagate(self.edge_index, x=x,
                                               edge_attr=edge_attr)

        # Use aggr_mlp only for grid nodes, mesh nodes have no aggregated messages input
        grid_x = x[:, self.N_mesh:]  # (B, N_grid, d_h)
        grid_rep = self.aggr_mlp(torch.cat((grid_x, grid_edge_rep_aggr), dim=-1))

        # Residual connection, only for grid nodes
        grid_rep = grid_rep + grid_x

        return grid_rep


class MeshInitNet(InteractionNet):
    """
    InteractionNet used in the mesh init step of hierarchical model

    The representations given as x here are [mesh_nodes_level_l-1, mesh_nodes_level_l]
    and edge_index should respect this, with index 0 being the first node in level l-1
    """

    def __init__(self, edge_index, edge_mlp, aggr_mlp, N_from_nodes, N_to_nodes):
        super().__init__(edge_index, edge_mlp, aggr_mlp)
        self.N_from_nodes = N_from_nodes
        self.N_to_nodes = N_to_nodes

    def aggregate(self, messages, index, ptr, dim_size):
        # Force to only aggregate to grid nodes
        shifted_index = index - self.N_from_nodes
        # Shift index so that top level l gets index 0
        return super().aggregate(messages, shifted_index, ptr, self.N_to_nodes)

    def forward(self, x, edge_attr):
        """
        x: (B, N_mesh[l-1]+N_mesh[l], d_h)
        edge_attr: (B, M_up[l-1 -> l], d_h)
        """
        self.edge_index = self.edge_index.to(x.device)

        top_level_aggr, edge_rep = self.propagate(self.edge_index, x=x,
                                                  edge_attr=edge_attr)

        # Use aggr_mlp only for mesh nodes at top level l
        top_level_x = x[:, self.N_from_nodes:]  # (B, N_mesh[l], d_h)
        top_level_rep = self.aggr_mlp(torch.cat((top_level_x, top_level_aggr), dim=-1))

        # Residual connections
        top_level_rep = top_level_rep + top_level_x
        edge_rep = edge_rep + edge_attr

        return top_level_rep, edge_rep


class MeshReadOutNet(InteractionNet):
    """
    InteractionNet used in read-out step of hierarchical model
    """

    def __init__(self, edge_index, edge_mlp, aggr_mlp, N_to_nodes):
        super().__init__(edge_index, edge_mlp, aggr_mlp)
        self.N_to_nodes = N_to_nodes

    def aggregate(self, messages, index, ptr, dim_size):
        # Force to only aggregate to lower level nodes
        return super().aggregate(messages, index, ptr, self.N_to_nodes)

    def forward(self, x, edge_attr):
        self.edge_index = self.edge_index.to(x.device)

        to_aggr, _ = self.propagate(self.edge_index, x=x,
                                    edge_attr=edge_attr)

        # Use aggr_mlp only for to nodes, from nodes have no aggregated messages input
        to_x = x[:, :self.N_to_nodes]  # (B, N_mesh[l], d_h)
        to_rep = self.aggr_mlp(torch.cat((to_x, to_aggr), dim=-1))

        # Residual connection, only for grid nodes
        to_rep = to_rep + to_x
        return to_rep


class MeshDownNet(InteractionNet):
    """
    InteractionNet used in downeward step of vertical hierarchical model
    Note: Same as MeshReadOutNet, but adds residual connection and returns also
    edge representations
    """

    def __init__(self, edge_index, edge_mlp, aggr_mlp, N_to_nodes):
        super().__init__(edge_index, edge_mlp, aggr_mlp)
        self.N_to_nodes = N_to_nodes

    def aggregate(self, messages, index, ptr, dim_size):
        # Force to only aggregate to lower level nodes
        return super().aggregate(messages, index, ptr, self.N_to_nodes)

    def forward(self, x, edge_attr):
        self.edge_index = self.edge_index.to(x.device)

        to_aggr, edge_rep = self.propagate(self.edge_index, x=x,
                                           edge_attr=edge_attr)

        # Use aggr_mlp only for to nodes, from nodes have no aggregated messages input
        to_x = x[:, :self.N_to_nodes]  # (B, N_mesh[l], d_h)
        to_rep = self.aggr_mlp(torch.cat((to_x, to_aggr), dim=-1))

        # Residual connection, both for grid nodes and edges
        to_rep = to_rep + to_x
        edge_rep = edge_rep + edge_attr
        return to_rep, edge_rep


class HiInteractionNet(InteractionNet):
    """
    InteractionNet used in processing step of hierarchical model
    Note that we do not have to keep track of which splitting of edges is between
    levels and between edge directions.
    """

    def __init__(self, edge_index, edge_mlp, aggr_mlp, edge_split_sections,
                 node_split_sections, aggr="sum"):
        super().__init__(edge_index, edge_mlp, aggr_mlp, aggr="sum")
        # Note that in this class edge_mlp and aggr_mlp are lists of mlp
        # Overwrite to put in ModuleLists
        self.edge_mlp = torch.nn.ModuleList(edge_mlp)
        self.aggr_mlp = torch.nn.ModuleList(aggr_mlp)

        # Lists of section lengths for splitting messages (edges) and nodes
        # Splitting is done to use separate MLP for each section
        self.edge_split_sections = edge_split_sections
        self.node_split_sections = node_split_sections

    def forward(self, x, edge_attr):
        """
        x: (B, N_mesh, d_h)
        edge_attr: (B, M_up + M_down + M_same, d_h)
        """
        self.edge_index = self.edge_index.to(x.device)

        edge_rep_aggr, edge_rep = self.propagate(self.edge_index, x=x,
                                                 edge_attr=edge_attr)

        aggr_input = torch.cat((x, edge_rep_aggr), dim=-1)  # (B, N_mesh, 2d_h)
        aggr_input_sections = torch.split(aggr_input, self.node_split_sections, dim=1)
        node_rep_sections = [mlp(section_input) for mlp, section_input in
                             zip(self.aggr_mlp, aggr_input_sections)]
        node_rep = torch.cat(node_rep_sections, dim=1)

        # Residual connections
        node_rep = node_rep + x
        edge_rep = edge_rep + edge_attr

        return node_rep, edge_rep

    def message(self, x_j, x_i, edge_attr):
        """
        Message fromm node j to node i.
        """
        mlp_input = torch.cat((edge_attr, x_j, x_i), dim=-1)
        # Split up messages, as different MLPs should be applied
        mlp_input_sections = torch.split(mlp_input, self.edge_split_sections, dim=1)
        messages_sections = [mlp(section_input) for mlp, section_input in
                             zip(self.edge_mlp, mlp_input_sections)]

        # Put back together for aggregation
        return torch.cat(messages_sections, dim=1)
