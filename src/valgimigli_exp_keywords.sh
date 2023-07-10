#! /bin/bash

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project gnnqa_keywords --dont_save --skip_test --load_dataset_from dataset/eli5_1000_1000_1000_conceptnet \
--val_samples 10 --run_info keybert

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project gnnqa_keywords --dont_save --skip_test --load_dataset_from dataset/eli5_1000_1000_1000_conceptnet_rake \
--val_samples 10 --run_info rake

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project gnnqa_keywords --dont_save --skip_test --load_dataset_from dataset/eli5_1000_1000_1000_conceptnet_yake \
--val_samples 10 --run_info yake