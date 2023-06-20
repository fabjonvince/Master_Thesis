#!/bin/bash
docker run -v $PWD:/workspace --rm -it --gpus device=$CUDA_VISIBLE_DEVICES qagnn $1