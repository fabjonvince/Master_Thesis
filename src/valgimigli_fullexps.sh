./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 5.2e-05 \
--project full_train --load_dataset_from dataset/aquamuse_6000_50_800_conceptnet_rake/ \
--val_samples 20 --run_info bart_base_rEvelio_aguamuse --gnn_topk 3 \
 --model_method bart --max_epochs 5 --dataset aquamuse --no_wandb --checkpoint_summarizer facebook/bart-base

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 5.2e-05 \
--project full_train --load_dataset_from dataset/aquamuse_6000_50_800_conceptnet_rake/ \
--val_samples 20 --run_info bart_large_rEvelio_aguamuse --gnn_topk 3 \
 --model_method bart --max_epochs 5 --checkpoint_summarizer facebook/bart-large --dataset aquamuse --no_wandb