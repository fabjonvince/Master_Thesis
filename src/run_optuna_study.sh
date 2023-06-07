#!/bin/bash

# Step 1: Check mysql installation and connectivity
if ! mysql -u optuna -e "exit" >/dev/null 2>&1; then
    echo "MySQL is not installed or not reachable using the 'optuna' user."
    exit 1
fi

# Step 2: Create the table in MySQL
mysql -u optuna -e "CREATE TABLE $1 (id INT AUTO_INCREMENT PRIMARY KEY, data JSON);" || {
    echo "Error creating table $1 in MySQL."
    exit 1
}

# Step 3: Check study_name field in config file
config_study_name=$(jq -r '.study_name' "$2")
if [[ "$config_study_name" != "$1" ]]; then
    echo "Warning: 'study_name' field in the config file is not equal to $1."
fi

# Step 4: Run python script using slurm on deeplearn2 node
sbatch --nodelist=deeplearn2 --gres=gpu:"$CUDA_VISIBLE_DEVICES" --wrap="docker run --gpus '\"device=$CUDA_VISIBLE_DEVICES\"' -v $(pwd):/workspace gnnqa python optuna_integration.py $2"
