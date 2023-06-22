#!/bin/bash

# usage ./run_optuna_study_using_slurm.sh config.json

config_file=$1

# check if config file exists
if [ ! -f $config_file ]; then
    echo "Config file not found!"
    exit 1
fi

# check if config file is valid json
python -c "import json; json.load(open('$config_file'))"

# extract from the config file the parameters: number_of_trails, study_name, mysql_server_url
number_of_parallel_jobs=$(jq -r '.number_of_parallel_jobs' "$config_file")
study_name=$(jq -r '.study_name' "$config_file")
mysql_server_url=$(jq -r '.mysql_server_url' "$config_file")

# print info
echo "Running $number_of_parallel_jobs parallel jobs for study $study_name"
echo "MySQL server url: $mysql_server_url"

./run_on_docker.sh "optuna create-study --study-name $study_name --storage \"$mysql_server_url$study_name\""
echo "Study created"


for i in $(seq 1 $number_of_parallel_jobs); do
  ./run_on_sbatch.sh "python optuna_integration.py $config_file"
done


