#!/bin/bash

# Get the node name where the script is running
node_name=$(hostname)

echo $node_name

if [[ "$node_name" == "deeplearn2" ]]; then
    gpu_name="titan_xp"
else
    gpu_name="nvidia_geforce_3090"
    exit 1
fi

# Construct the sbatch command with the selected node name and GPU
sbatch -N 1 --gpus="$gpu_name" -w "$node_name" run_on_docker.sh "$@"
