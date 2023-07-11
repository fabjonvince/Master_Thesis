
# t5
./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/eli5_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_base_revelio_eli5 --topk 3 --train_samples 5000 --model_method t5 --max_epochs 5

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/eli5_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_base_eli5 --train_samples 5000 --model_method t5 --max_epochs 5

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/eli5_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_large_revelio_eli5 --topk 3 --train_samples 5000 --model_method t5 --max_epochs 5 \
--checkpoint_summarizer t5-large

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/eli5_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_large_eli5 --train_samples 5000 --model_method t5 --max_epochs 5 \
--checkpoint_summarizer t5-large


# bart
./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/eli5_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_base_revelio_eli5 --topk 3 --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-base

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/eli5_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_base_eli5 --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-base

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/eli5_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_large_revelio_eli5 --topk 3 --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-large

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/eli5_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_large_eli5 --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-large


#aquamuse

# t5
./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/aquamuse_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_base_revelio_aquamuse --topk 3 --train_samples 5000 --model_method t5 --max_epochs 5

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/aquamuse_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_base_aquamuse --train_samples 5000 --model_method t5 --max_epochs 5

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/aquamuse_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_large_revelio_aquamuse --topk 3 --train_samples 5000 --model_method t5 --max_epochs 5 \
--checkpoint_summarizer t5-large

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/aquamuse_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_large_aquamuse --train_samples 5000 --model_method t5 --max_epochs 5 \
--checkpoint_summarizer t5-large


# bart
./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/aquamuse_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_base_revelio_aquamuse --topk 3 --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-base

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/aquamuse_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_base_aquamuse --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-base

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/aquamuse_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_large_revelio_aquamuse --topk 3 --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-large

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/aquamuse_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_large_aquamuse --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-large

#msmarco (din0s)

# t5
./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/din0s_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_base_revelio_msmarco --topk 3 --train_samples 5000 --model_method t5 --max_epochs 5

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/din0s_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_base_msmarco --train_samples 5000 --model_method t5 --max_epochs 5

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/din0s_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_large_revelio_msmarco --topk 3 --train_samples 5000 --model_method t5 --max_epochs 5 \
--checkpoint_summarizer t5-large

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/din0s_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info t5_large_msmarco --train_samples 5000 --model_method t5 --max_epochs 5 \
--checkpoint_summarizer t5-large


# bart
./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/din0s_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_base_revelio_msmarco --topk 3 --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-base

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/din0s_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_base_msmarco --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-base

./run_on_sbatch.sh python main.py --layer_with_gnn 0 3 7 --model_lr 2.5e-05 --gnn_lr 1.2e-06 \
--wandb_project full_train --load_dataset_from dataset/din0s_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_large_revelio_msmarco --topk 3 --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-large

./run_on_sbatch.sh python main.py --no_gnn --model_lr 2.5e-05 \
--wandb_project full_train --load_dataset_from dataset/din0s_10000_50_1000_conceptnet_rake \
--val_samples 10 --run_info bart_large_msmarco --train_samples 5000 --model_method bart --max_epochs 5 \
--checkpoint_summarizer facebook/bart-large