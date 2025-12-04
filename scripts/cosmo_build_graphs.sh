#!/bin/sh

CP=${1:-cosmo_model_config_era5.yaml}

# Rectangular
# Mesh node distance in COSMO rotated pole CRS
MND=0.1
python -m neural_lam.build_rectangular_graph \
    --config_path "$CP" \
    --mesh_node_distance "$MND" \
    --archetype hierarchical \
    --max_num_levels 3 \
    --graph_name rectangular_hierarchical
