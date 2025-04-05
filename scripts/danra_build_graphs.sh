#!/bin/bash

CP=danra_model_config_era5.yaml

# Rectangular
MND=12500

# hierarchical
for lev in {2..4}; do
    python -m neural_lam.build_rectangular_graph --config_path $CP --mesh_node_distance $MND --archetype hierarchical --max_num_levels $lev --graph_name rect_hi$lev
done

# multiscale3
python -m neural_lam.build_rectangular_graph --config_path $CP --mesh_node_distance $MND --archetype graphcast --graph_name rect_ms3 --max_num_levels 3

# multiscale4
python -m neural_lam.build_rectangular_graph --config_path $CP --mesh_node_distance $MND --archetype graphcast --graph_name rect_ms4 --max_num_levels 4

# Triangular
SPLITS=9
RB=1.85

# hierarchical
for lev in {3..4}; do
    python -m neural_lam.build_triangular_graph --config_path $CP --hierarchical --splits $SPLITS --levels $lev --graph_name tri_9s_hi$lev --rotate_ico --g2m_radius_boundary $RB --two_dim_features
done

# multiscale3
python -m neural_lam.build_triangular_graph --config_path $CP --splits $SPLITS --levels 3 --graph_name tri_9s_ms3 --rotate_ico --g2m_radius_boundary $RB --two_dim_features
