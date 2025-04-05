#!/bin/sh

CP=cosmo_model_config_era5.yaml

# Rectangular
# Mesh node distance in COSMO rotated pole CRS
MND=0.1
python -m neural_lam.build_rectangular_graph --config_path $CP --mesh_node_distance $MND --archetype hierarchical --max_num_levels 3 --graph_name rectangular_hierarchical

# Triangular
python -m neural_lam.build_triangular_graph --config_path $CP --graph_name triangular_hierarchical --hierarchical --splits 9 --levels 3
