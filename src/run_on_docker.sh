#!/bin/bash
docker run -v $PWD:/workspace --rm --gpus device=$CUDA_VISIBLE_DEVICES qagnn $*