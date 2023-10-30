#!/bin/bash

# Define the parameters and levels
params=("T" "U" "V" "RELHUM")
levels=(1 5 13 22 38 41 60)

# Get the output directory from the first script argument
output_dir=$1

# Loop over all combinations
for param in "${params[@]}"; do
    for level in "${levels[@]}"; do
        # Generate the GIF
        convert -delay 20 -loop 0 ${output_dir}/${param}_lvl_${level}_t_* ${output_dir}/${param}_lvl_${level}.gif
    done
done
