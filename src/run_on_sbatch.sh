#!/usr/bin/env bash

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_on_docker.sh $*