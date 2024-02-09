# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import logging

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from graphcast_utils import (
    add_edge_features,
    add_node_features,
    cell_to_adj,
    create_graph,
    create_heterograph,
    get_edge_len,
    latlon2xyz,
    find_subset_indices,
)

logger = logging.getLogger(__name__)


class Graph:
    """Graph class for creating the graph2mesh, multimesh, and mesh2graph graphs.

    Parameters
    ----------
    icospheres_path : str
        Path to the icospheres json file.
        If the file does not exist, it will try to generate it using PyMesh.
    lat_lon_grid : Tensor
        Tensor with shape (lat, lon, 2) that includes the latitudes and longitudes
        meshgrid.
    dtype : torch.dtype, optional
        Data type of the graph, by default torch.float
        
        
    Icospheres Doc
    --------------
    The icospheres json file is a dictionary with the following keys:
    - order_0_vertices: np.ndarray
        (n_vertices, 3)
        The vertices of the icosphere with order 0.
        Each row is the xyz coordinates of a vertex.
    - order_0_faces: np.ndarray 
        (n_faces, 3)
        The faces of the icosphere with order 0.
        Each row contains 3 indices to the order_0_vertices which make up the triangular face.
    - order_0_face_centroid: np.ndarray
        (n_faces, 3)
        The centroid of each face of the icosphere with order 0.
        The xyz coordinates of each centroid.
    """

    def __init__(
        self, icospheres_path: str, lat_lon_grid: Tensor, local_lat_lon_grid=None, dtype=torch.float
    ) -> None:
        self.dtype = dtype
        # Get or generate the icospheres
        try:
            with open(icospheres_path, "r") as f:
                loaded_dict = json.load(f)
                icospheres = {
                    key: (np.array(value) if isinstance(value, list) else value)
                    for key, value in loaded_dict.items()
                }
                print(f"Opened pre-computed graph at {icospheres_path}.")
        except FileNotFoundError:
            print(
                f"Could not open {icospheres_path}...generating mesh from scratch."
            )

        # icospheres
        # 
        self.icospheres = icospheres
        self.max_order = (
            len([key for key in self.icospheres.keys() if "faces" in key]) - 2
        )

        # flatten lat/lon gird
        # (lat, lon, 2) -> (2, lat, lon) -> (2, lat*lon) -> (lat*lon, 2)
        self.lat_lon_grid_flat = lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0) # (lat*lon, 2)
        self.local_lat_lon_grid_flat = local_lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0) # (lat*lon, 2)
        self.local2global_idxs = find_subset_indices(self.lat_lon_grid_flat, self.local_lat_lon_grid_flat)
        self.local2global_idxs_set = set(self.local2global_idxs)
        
    def create_mesh_graph(self, verbose: bool = True) -> Tensor:
        """Create the multimesh graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        DGLGraph
            Multimesh graph.
        """
        # create the bi-directional mesh graph
        multimesh_faces = self.icospheres["order_0_faces"]
        for i in range(1, self.max_order + 1):
            multimesh_faces = np.concatenate(
                (multimesh_faces, self.icospheres["order_" + str(i) + "_faces"])
            )

        src, dst = cell_to_adj(multimesh_faces)
        # edge subset
        src_subset, dst_subset = [], []
        for i in range(len(src)):
            if src[i] in self.g2m_node_subset_set and dst[i] in self.g2m_node_subset_set:
                src_subset.append(src[i])
                dst_subset.append(dst[i])
        src = [self.g2m_node_subset.index(i) for i in src_subset]
        dst = [self.g2m_node_subset.index(i) for i in dst_subset]
        # node subset
        mesh_features = [self.icospheres["order_" + str(self.max_order) + "_vertices"][i] for i in self.g2m_node_subset]
        mesh_graph = create_graph(
            src, dst, to_bidirected=True, add_self_loop=False, dtype=torch.int32
        )
        mesh_pos = torch.tensor(
            mesh_features,
            dtype=torch.float32,
        )
        mesh_graph = add_edge_features(mesh_graph, mesh_pos)
        mesh_graph = add_node_features(mesh_graph, mesh_pos)
        # ensure fields set to dtype to avoid later conversions
        mesh_graph.ndata["x"] = mesh_graph.ndata["x"].to(dtype=self.dtype)
        mesh_graph.edata["x"] = mesh_graph.edata["x"].to(dtype=self.dtype)
        if verbose:
            print("mesh graph:", mesh_graph)
        return mesh_graph

    def create_g2m_graph(self, verbose: bool = True) -> Tensor:
        """Create the graph2mesh graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        DGLGraph
            Graph2mesh graph.
        """
        # get the max edge length of icosphere with max order
        edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 0]
        ]
        edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 1]
        ]
        edge_len_1 = np.max(get_edge_len(edge_src, edge_dst))
        edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 0]
        ]
        edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 2]
        ]
        edge_len_2 = np.max(get_edge_len(edge_src, edge_dst))
        edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 1]
        ]
        edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 2]
        ]
        edge_len_3 = np.max(get_edge_len(edge_src, edge_dst))
        edge_len = max([edge_len_1, edge_len_2, edge_len_3])

        # create the grid2mesh bipartite graph
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat) # (lat*lon, 3)
        n_nbrs = 4
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
            self.icospheres["order_" + str(self.max_order) + "_vertices"] # (40962, 3)
        )
        distances, indices = neighbors.kneighbors(cartesian_grid) # (lat*lon, n_nbrs)
        
        src, dst = [], []
        for i in range(len(cartesian_grid)):
            for j in range(n_nbrs):
                if distances[i][j] <= 0.6 * edge_len:
                    src.append(i)
                    dst.append(indices[i][j])
                    # NOTE this gives 1,624,344 edges, in the paper it is 1,618,746
                    # this number is very sensitive to the chosen edge_len, not clear
                    # in the paper what they use.
        
        # get edge subsets
        src_subset = [s for s in src if s in self.local2global_idxs_set]
        dst_subset = [dst[i] for i in range(len(src)) if src[i] in self.local2global_idxs_set]
        
        # subgraph indices
        self.g2m_node_subset = sorted(list(set(dst_subset)))
        self.g2m_node_subset_set = set(dst_subset)
        
        # reindex the dst to match the new node indices
        src = [self.local2global_idxs.index(i) for i in src_subset]
        dst = [self.g2m_node_subset.index(i) for i in dst_subset]
    
        # subsets of features
        cartesian_grid = cartesian_grid[self.local2global_idxs]
        mesh_node_features = [self.icospheres["order_" + str(self.max_order) + "_vertices"][i] for i in self.g2m_node_subset]
        
        g2m_graph = create_heterograph(
            src, dst, ("grid", "g2m", "mesh"), dtype=torch.int32
        )
        g2m_graph.srcdata["pos"] = cartesian_grid.to(torch.float32)
        g2m_graph.dstdata["pos"] = torch.tensor(
            mesh_node_features,
            dtype=torch.float32,
        )
        g2m_graph = add_edge_features(
            g2m_graph, (g2m_graph.srcdata["pos"], g2m_graph.dstdata["pos"])
        )
        # avoid potential conversions at later points
        g2m_graph.srcdata["pos"] = g2m_graph.srcdata["pos"].to(dtype=self.dtype)
        g2m_graph.dstdata["pos"] = g2m_graph.dstdata["pos"].to(dtype=self.dtype)
        g2m_graph.ndata["pos"]["grid"] = g2m_graph.ndata["pos"]["grid"].to(
            dtype=self.dtype
        )
        g2m_graph.ndata["pos"]["mesh"] = g2m_graph.ndata["pos"]["mesh"].to(
            dtype=self.dtype
        )
        g2m_graph.edata["x"] = g2m_graph.edata["x"].to(dtype=self.dtype)
        if verbose:
            print("g2m graph:", g2m_graph)
        return g2m_graph

    def create_m2g_graph(self, verbose: bool = True) -> Tensor:
        """Create the mesh2grid graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        DGLGraph
            Mesh2grid graph.
        """
        # create the mesh2grid bipartite graph
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        n_nbrs = 1
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
            self.icospheres["order_" + str(self.max_order) + "_face_centroid"]
        )
        _, indices = neighbors.kneighbors(cartesian_grid) # find the closest face centroid for each grid point (lat*lon, 1)
        indices = indices.flatten() # (lat*lon,) 
        
        # get edge subsets
        # src_subset = [
        #     p
        #     for i in indices[self.local2global_idxs]
        #     for p in self.icospheres["order_" + str(self.max_order) + "_faces"][i]
        # ]
        # dst_subset = [i for i in self.local2global_idxs for _ in range(3)]
        src_subset, dst_subset = [], []
        for grid_idx, mesh_idx in enumerate(indices[self.local2global_idxs]):
            for p in self.icospheres["order_" + str(self.max_order) + "_faces"][mesh_idx]:
                if p in self.g2m_node_subset_set:
                    src_subset.append(p)
                    dst_subset.append(grid_idx)
                    
        # reindex edges
        src = [self.g2m_node_subset.index(i) for i in src_subset]
        dst = dst_subset
        
        # subsets of features
        cartesian_grid = cartesian_grid[self.local2global_idxs]
        mesh_node_features = [self.icospheres["order_" + str(self.max_order) + "_vertices"][i] for i in self.g2m_node_subset]
        
        m2g_graph = create_heterograph(
            src, dst, ("mesh", "m2g", "grid"), dtype=torch.int32
        )  # number of edges is 3,114,720, exactly matches with the paper
        m2g_graph.srcdata["pos"] = torch.tensor(
            mesh_node_features,
            dtype=torch.float32,
        )
        m2g_graph.dstdata["pos"] = cartesian_grid.to(dtype=torch.float32)
        m2g_graph = add_edge_features(
            m2g_graph, (m2g_graph.srcdata["pos"], m2g_graph.dstdata["pos"])
        )
        # avoid potential conversions at later points
        m2g_graph.srcdata["pos"] = m2g_graph.srcdata["pos"].to(dtype=self.dtype)
        m2g_graph.dstdata["pos"] = m2g_graph.dstdata["pos"].to(dtype=self.dtype)
        m2g_graph.ndata["pos"]["grid"] = m2g_graph.ndata["pos"]["grid"].to(
            dtype=self.dtype
        )
        m2g_graph.ndata["pos"]["mesh"] = m2g_graph.ndata["pos"]["mesh"].to(
            dtype=self.dtype
        )
        m2g_graph.edata["x"] = m2g_graph.edata["x"].to(dtype=self.dtype)

        if verbose:
            print("m2g graph:", m2g_graph)
        return m2g_graph
    
    def find_subset_graphs(self, verbose: bool = True):
        # ===== Get G2M Mesh Subset ===== #
        # get the max edge length of icosphere with max order
        edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 0]
        ]
        edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 1]
        ]
        edge_len_1 = np.max(get_edge_len(edge_src, edge_dst))
        edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 0]
        ]
        edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 2]
        ]
        edge_len_2 = np.max(get_edge_len(edge_src, edge_dst))
        edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 1]
        ]
        edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 2]
        ]
        edge_len_3 = np.max(get_edge_len(edge_src, edge_dst))
        edge_len = max([edge_len_1, edge_len_2, edge_len_3])

        # create the grid2mesh bipartite graph
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat) # (lat*lon, 3)
        n_nbrs = 4
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
            self.icospheres["order_" + str(self.max_order) + "_vertices"] # (40962, 3)
        )
        distances, indices = neighbors.kneighbors(cartesian_grid) # (lat*lon, n_nbrs)
        
        src, dst = [], []
        for i in range(len(cartesian_grid)):
            for j in range(n_nbrs):
                if distances[i][j] <= 0.6 * edge_len:
                    src.append(i)
                    dst.append(indices[i][j])
                    # NOTE this gives 1,624,344 edges, in the paper it is 1,618,746
                    # this number is very sensitive to the chosen edge_len, not clear
                    # in the paper what they use.
        
        g2m_mesh_subset = [dst[i] for i in range(len(src)) if src[i] in self.local2global_idxs]
        
        # ===== Get M2G Mesh Subset ===== #
        # create the mesh2grid bipartite graph
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        n_nbrs = 1
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
            self.icospheres["order_" + str(self.max_order) + "_face_centroid"]
        )
        _, indices = neighbors.kneighbors(cartesian_grid) # find the closest face centroid for each grid point (lat*lon, 1)
        mesh_indices = indices.flatten() # (lat*lon,)
        
        m2g_mesh_subset = [
            p
            for i in indices[self.local2global_idxs]
            for p in self.icospheres["order_" + str(self.max_order) + "_faces"][i]
        ]
        dst_subset = [i for i in self.local2global_idxs for _ in range(3)]
        pass

def create_graphcast_mesh(args):
    graph_dir_path = os.path.join("graphs", args.graph)
    os.makedirs(graph_dir_path, exist_ok=True)
    
    data_dir_path = os.path.join("data", args.dataset)
    icosophere_path = os.path.join(data_dir_path, "icospheres.json")
    
    # Get or generate lat_lon_grid
    # nwp_xy_path = os.path.join(data_dir_path, "static", "nwp_xy.npy")
    # try:
    #     # with open(nwp_xy_path, "r") as f:
    #     lat_lon_grid = np.load(nwp_xy_path) # (2, lon, lat) or (2, x, y)
    #     lat_lon_grid = torch.from_numpy(lat_lon_grid).permute(2, 1, 0) # (lat, lon, 2)
    #     print(lat_lon_grid.shape)
    #     print(f"Opened lat lon grid at {nwp_xy_path}.")
    # except FileNotFoundError:
    #     # create the lat_lon_grid
    #     print(
    #         f"Could not open {nwp_xy_path}...generating lat lon grid from scratch."
    #     )
    #     input_res = (721, 1440) # (lat, lon)
    #     latitudes = torch.linspace(-90, 90, steps=input_res[0])
    #     longitudes = torch.linspace(-180, 180, steps=input_res[1] + 1)[1:]
    #     lat_lon_grid = torch.stack(
    #         torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1
    #     ) # (lat, lon, 2)
    #     print(lat_lon_grid.shape)

    nwp_xy_path = os.path.join(data_dir_path, "static", "nwp_xy.npy")
    local_lat_lon_grid = np.load(nwp_xy_path) # (2, lon, lat) or (2, x, y)
    local_lat_lon_grid = torch.from_numpy(local_lat_lon_grid).permute(2, 1, 0) # (lat, lon, 2)
    print(local_lat_lon_grid.shape)
    print(f"Opened lat lon grid at {nwp_xy_path}.")
    
    input_res = (721, 1440) # (lat, lon)
    latitudes = torch.linspace(-90, 90, steps=input_res[0])
    longitudes = torch.linspace(-180, 180, steps=input_res[1] + 1)[1:]
    lat_lon_grid = torch.stack(
        torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1
    ) # (lat, lon, 2)
    print(lat_lon_grid.shape)

    graph = Graph(icosophere_path, lat_lon_grid, local_lat_lon_grid)
    
    # graph.find_subset_graphs()
    
    # ----- Grid2Mesh Graph ----- #
    g2m_graph = graph.create_g2m_graph()
    src, dst = g2m_graph.edges()
    g2m_edge_index = torch.stack((src, dst)).to(torch.int64)
    torch.save(g2m_edge_index, 
               os.path.join(graph_dir_path, f"g2m_edge_index.pt"))
    torch.save(g2m_graph.edata["x"], 
               os.path.join(graph_dir_path, f"g2m_features.pt"))
    
    # ----- Mesh2Grid Graph ----- #
    m2g_graph = graph.create_m2g_graph()
    src, dst = m2g_graph.edges()
    m2g_edge_index = torch.stack((src, dst)).to(torch.int64)
    torch.save(m2g_edge_index, 
               os.path.join(graph_dir_path, f"m2g_edge_index.pt"))
    torch.save(m2g_graph.edata["x"], 
               os.path.join(graph_dir_path, f"m2g_features.pt"))
    
    # ----- Mesh Graph ----- #
    mesh_graph = graph.create_mesh_graph()
    src, dst = mesh_graph.edges()
    m2m_edge_index = torch.stack((src, dst)).to(torch.int64)
    torch.save([m2m_edge_index], 
               os.path.join(graph_dir_path, f"m2m_edge_index.pt"))
    torch.save([mesh_graph.edata["x"]], 
               os.path.join(graph_dir_path, f"m2m_features.pt"))
    torch.save([mesh_graph.ndata["x"]], 
               os.path.join(graph_dir_path, f"mesh_features.pt"))

# class Graph:
#     """Graph class for creating the graph2mesh, multimesh, and mesh2graph graphs.

#     Parameters
#     ----------
#     icospheres_path : str
#         Path to the icospheres json file.
#         If the file does not exist, it will try to generate it using PyMesh.
#     lat_lon_grid : Tensor
#         Tensor with shape (lat, lon, 2) that includes the latitudes and longitudes
#         meshgrid.
#     dtype : torch.dtype, optional
#         Data type of the graph, by default torch.float
#     """

#     def __init__(
#         self, icospheres_path: str, lat_lon_grid: Tensor, dtype=torch.float
#     ) -> None:
#         self.dtype = dtype
#         # Get or generate the icospheres
#         try:
#             with open(icospheres_path, "r") as f:
#                 loaded_dict = json.load(f)
#                 icospheres = {
#                     key: (np.array(value) if isinstance(value, list) else value)
#                     for key, value in loaded_dict.items()
#                 }
#                 print(f"Opened pre-computed graph at {icospheres_path}.")
#         except FileNotFoundError:
#             print(
#                 f"Could not open {icospheres_path}...generating mesh from scratch."
#             )


#         self.icospheres = icospheres
#         self.max_order = (
#             len([key for key in self.icospheres.keys() if "faces" in key]) - 2
#         )

#         # flatten lat/lon gird
#         self.lat_lon_grid_flat = lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)

#     def create_mesh_graph(self, verbose: bool = True) -> Tensor:
#         """Create the multimesh graph.

#         Parameters
#         ----------
#         verbose : bool, optional
#             verbosity, by default True

#         Returns
#         -------
#         DGLGraph
#             Multimesh graph.
#         """
#         # create the bi-directional mesh graph
#         multimesh_faces = self.icospheres["order_0_faces"]
#         for i in range(1, self.max_order + 1):
#             multimesh_faces = np.concatenate(
#                 (multimesh_faces, self.icospheres["order_" + str(i) + "_faces"])
#             )

#         src, dst = cell_to_adj(multimesh_faces)
#         mesh_graph = create_graph(
#             src, dst, to_bidirected=True, add_self_loop=False, dtype=torch.int32
#         )
#         mesh_pos = torch.tensor(
#             self.icospheres["order_" + str(self.max_order) + "_vertices"],
#             dtype=torch.float32,
#         )
#         mesh_graph = add_edge_features(mesh_graph, mesh_pos)
#         mesh_graph = add_node_features(mesh_graph, mesh_pos)
#         # ensure fields set to dtype to avoid later conversions
#         mesh_graph.ndata["x"] = mesh_graph.ndata["x"].to(dtype=self.dtype)
#         mesh_graph.edata["x"] = mesh_graph.edata["x"].to(dtype=self.dtype)
#         if verbose:
#             print("mesh graph:", mesh_graph)
#         return mesh_graph

#     def create_g2m_graph(self, verbose: bool = True) -> Tensor:
#         """Create the graph2mesh graph.

#         Parameters
#         ----------
#         verbose : bool, optional
#             verbosity, by default True

#         Returns
#         -------
#         DGLGraph
#             Graph2mesh graph.
#         """
#         # get the max edge length of icosphere with max order
#         edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
#             self.icospheres["order_" + str(self.max_order) + "_faces"][:, 0]
#         ]
#         edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
#             self.icospheres["order_" + str(self.max_order) + "_faces"][:, 1]
#         ]
#         edge_len_1 = np.max(get_edge_len(edge_src, edge_dst))
#         edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
#             self.icospheres["order_" + str(self.max_order) + "_faces"][:, 0]
#         ]
#         edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
#             self.icospheres["order_" + str(self.max_order) + "_faces"][:, 2]
#         ]
#         edge_len_2 = np.max(get_edge_len(edge_src, edge_dst))
#         edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
#             self.icospheres["order_" + str(self.max_order) + "_faces"][:, 1]
#         ]
#         edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
#             self.icospheres["order_" + str(self.max_order) + "_faces"][:, 2]
#         ]
#         edge_len_3 = np.max(get_edge_len(edge_src, edge_dst))
#         edge_len = max([edge_len_1, edge_len_2, edge_len_3])

#         # create the grid2mesh bipartite graph
#         cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
#         n_nbrs = 4
#         neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
#             self.icospheres["order_" + str(self.max_order) + "_vertices"]
#         )
#         distances, indices = neighbors.kneighbors(cartesian_grid)

#         src, dst = [], []
#         for i in range(len(cartesian_grid)):
#             for j in range(n_nbrs):
#                 if distances[i][j] <= 0.6 * edge_len:
#                     src.append(i)
#                     dst.append(indices[i][j])
#                     # NOTE this gives 1,624,344 edges, in the paper it is 1,618,746
#                     # this number is very sensitive to the chosen edge_len, not clear
#                     # in the paper what they use.

#         g2m_graph = create_heterograph(
#             src, dst, ("grid", "g2m", "mesh"), dtype=torch.int32
#         )  # number of edges is 3,114,720, exactly matches with the paper
#         g2m_graph.srcdata["pos"] = cartesian_grid.to(torch.float32)
#         g2m_graph.dstdata["pos"] = torch.tensor(
#             self.icospheres["order_" + str(self.max_order) + "_vertices"],
#             dtype=torch.float32,
#         )
#         g2m_graph = add_edge_features(
#             g2m_graph, (g2m_graph.srcdata["pos"], g2m_graph.dstdata["pos"])
#         )
#         # avoid potential conversions at later points
#         g2m_graph.srcdata["pos"] = g2m_graph.srcdata["pos"].to(dtype=self.dtype)
#         g2m_graph.dstdata["pos"] = g2m_graph.dstdata["pos"].to(dtype=self.dtype)
#         g2m_graph.ndata["pos"]["grid"] = g2m_graph.ndata["pos"]["grid"].to(
#             dtype=self.dtype
#         )
#         g2m_graph.ndata["pos"]["mesh"] = g2m_graph.ndata["pos"]["mesh"].to(
#             dtype=self.dtype
#         )
#         g2m_graph.edata["x"] = g2m_graph.edata["x"].to(dtype=self.dtype)
#         if verbose:
#             print("g2m graph:", g2m_graph)
#         return g2m_graph

#     def create_m2g_graph(self, verbose: bool = True) -> Tensor:
#         """Create the mesh2grid graph.

#         Parameters
#         ----------
#         verbose : bool, optional
#             verbosity, by default True

#         Returns
#         -------
#         DGLGraph
#             Mesh2grid graph.
#         """
#         # create the mesh2grid bipartite graph
#         cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
#         n_nbrs = 1
#         neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
#             self.icospheres["order_" + str(self.max_order) + "_face_centroid"]
#         )
#         _, indices = neighbors.kneighbors(cartesian_grid)
#         indices = indices.flatten()

#         src = [
#             p
#             for i in indices
#             for p in self.icospheres["order_" + str(self.max_order) + "_faces"][i]
#         ]
#         dst = [i for i in range(len(cartesian_grid)) for _ in range(3)]
#         m2g_graph = create_heterograph(
#             src, dst, ("mesh", "m2g", "grid"), dtype=torch.int32
#         )  # number of edges is 3,114,720, exactly matches with the paper
#         m2g_graph.srcdata["pos"] = torch.tensor(
#             self.icospheres["order_" + str(self.max_order) + "_vertices"],
#             dtype=torch.float32,
#         )
#         m2g_graph.dstdata["pos"] = cartesian_grid.to(dtype=torch.float32)
#         m2g_graph = add_edge_features(
#             m2g_graph, (m2g_graph.srcdata["pos"], m2g_graph.dstdata["pos"])
#         )
#         # avoid potential conversions at later points
#         m2g_graph.srcdata["pos"] = m2g_graph.srcdata["pos"].to(dtype=self.dtype)
#         m2g_graph.dstdata["pos"] = m2g_graph.dstdata["pos"].to(dtype=self.dtype)
#         m2g_graph.ndata["pos"]["grid"] = m2g_graph.ndata["pos"]["grid"].to(
#             dtype=self.dtype
#         )
#         m2g_graph.ndata["pos"]["mesh"] = m2g_graph.ndata["pos"]["mesh"].to(
#             dtype=self.dtype
#         )
#         m2g_graph.edata["x"] = m2g_graph.edata["x"].to(dtype=self.dtype)

#         if verbose:
#             print("m2g graph:", m2g_graph)
#         return m2g_graph

# def create_graphcast_mesh(args):
#     graph_dir_path = os.path.join("graphs", args.graph)
#     os.makedirs(graph_dir_path, exist_ok=True)
    
#     data_dir_path = os.path.join("data", args.dataset)
#     icosophere_path = os.path.join(data_dir_path, "icospheres.json")
    
#     # create the lat_lon_grid
#     input_res = (721, 1440)
#     latitudes = torch.linspace(-90, 90, steps=input_res[0])
#     longitudes = torch.linspace(-180, 180, steps=input_res[1] + 1)[1:]
#     lat_lon_grid = torch.stack(
#         torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1
#     )
#     graph = Graph(icosophere_path, lat_lon_grid)
#     # ----- Mesh Graph ----- #
#     mesh_graph = graph.create_mesh_graph()
#     src, dst = mesh_graph.edges()
#     m2m_edge_index = torch.stack((src, dst)).to(torch.int64)
#     torch.save([m2m_edge_index], 
#                os.path.join(graph_dir_path, f"m2m_edge_index.pt"))
#     torch.save([mesh_graph.edata["x"]], 
#                os.path.join(graph_dir_path, f"m2m_features.pt"))
#     torch.save([mesh_graph.ndata["x"]], 
#                os.path.join(graph_dir_path, f"mesh_features.pt"))
    
#     # ----- Grid2Mesh Graph ----- #
#     g2m_graph = graph.create_g2m_graph()
#     src, dst = g2m_graph.edges()
#     g2m_edge_index = torch.stack((src, dst)).to(torch.int64)
#     torch.save(g2m_edge_index, 
#                os.path.join(graph_dir_path, f"g2m_edge_index.pt"))
#     torch.save(g2m_graph.edata["x"], 
#                os.path.join(graph_dir_path, f"g2m_features.pt"))
    
#     # ----- Mesh2Grid Graph ----- #
#     m2g_graph = graph.create_m2g_graph()
#     src, dst = m2g_graph.edges()
#     m2g_edge_index = torch.stack((src, dst)).to(torch.int64)
#     torch.save(m2g_edge_index, 
#                os.path.join(graph_dir_path, f"m2g_edge_index.pt"))
#     torch.save(m2g_graph.edata["x"], 
#                os.path.join(graph_dir_path, f"m2g_features.pt"))