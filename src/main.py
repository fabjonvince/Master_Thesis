import argparse
from datetime import datetime
import os
import time
import pdb

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, Dataset
from transformers import T5Tokenizer, TrainingArguments
import wandb
from preprocess import text_to_graph_concept, add_special_tokens, text_to_keywords, create_memory, \
    graph_to_nodes_and_rel, get_node_and_rel_dict
from data import get_dataset
from model import GNNQA
from t5 import T5GNNForConditionalGeneration
from pytorch_lightning import Trainer
from sentence_transformers import SentenceTransformer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from pytorch_lightning.loggers import WandbLogger

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='eli5', help='Dataset to use')
argparser.add_argument('--run_info', type=str, default=None, help='Run info that will be added to run name')
argparser.add_argument('--wandb_project', type=str, default='gnnqa_default_project', help='wandb project name')
argparser.add_argument('--graph', type=str, default='conceptnet', help='Graph to use')
argparser.add_argument('--train_samples', type=int, default=10, help='Number of train samples')
argparser.add_argument('--val_samples', type=int, default=10, help='Number of validation samples')
argparser.add_argument('--test_samples', type=int, default=10, help='Number of test samples')
argparser.add_argument('--accumulate_grad_batches', type=int, default=8,
                       help='Number of batches to accumulate gradients')
argparser.add_argument('--layer_with_gnn', type=int, default=[1, 2], help='Layers with KIL')
argparser.add_argument('--debug', action='store_true', help='Debug mode')
argparser.add_argument('--gnn_topk', type=int, default=2, help='Number of topk nodes to consider for each root node')
argparser.add_argument('--load_dataset_from', type=str, default=None, help='Load dataset from path')
argparser.add_argument('--checkpoint_sentence_transformer', type=str, default='all-MiniLM-L12-v2',
                       help='Load sentence transformer checkpoint')
argparser.add_argument('--checkpoint_summarizer', type=str, default='t5-base',
                       help='Summarizer checkpoint from huggingface')
argparser.add_argument('--sentence_transformer_embedding_size', type=int, default=384,
                       help='Sentence transformer embedding size')
argparser.add_argument('--max_length', type=int, default=512, help='Max length of the input sequence')
argparser.add_argument('--graph_depth', type=int, default=3, help='Graph depth')
argparser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
argparser.add_argument('--max_epochs', type=int, default=1, help='max number of epochs')
argparser.add_argument('--save_top_k', type=int, default=1, help='save top k checkpoints')
argparser.add_argument('--dont_save', default=False, action='store_true', help='do not save the model')
argparser.add_argument('--no_wandb', default=False, action='store_true', help='do not use wandb')
argparser.add_argument('--no_gnn', default=False, action='store_true', help='do not use gnn')
argparser.add_argument('--optuna_pruner_callback', default=None, help='optuna pruner callback')
argparser.add_argument('--skip_test', default=False, action='store_true', help='skip test')
argparser.add_argument('--skip_train', default=False, action='store_true', help='skip train')
argparser.add_argument('--model_lr', default=0.000001, type=float, help='model learning rate')
argparser.add_argument('--gnn_lr', default=None, type=float, help='gnn learning rate')
name_mapping = {
    "eli5": ("train_eli5", "validation_eli5", "test_eli5", "title", "answers"),
    "conceptnet": ("rel", "arg1", "arg2"),
}


def main(args):
    dataset_columns = name_mapping.get(args.dataset, None)
    train_name = dataset_columns[0]
    eval_name = dataset_columns[1]
    test_name = dataset_columns[2]
    question_name = dataset_columns[3]
    answers_name = dataset_columns[4]
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    tokenizer.add_special_tokens(
        {"additional_special_tokens": tokenizer.additional_special_tokens + ["<REL_TOK>", "<GNN_TOK>"]})
    nodes, rels = get_node_and_rel_dict()
    # load dataset
    if args.load_dataset_from is not None:

        dataset = load_from_disk(args.load_dataset_from)

    else:

        save_dir = f'dataset/eli5_{args.train_samples}_{args.val_samples}_{args.test_samples}_conceptnet'
        dataset = get_dataset(args.dataset)

        nodes_dict = {row.custom_value: row.custom_index for row in nodes.itertuples(index=True)}
        rels_dict = {row.custom_value: row.custom_index for row in rels.itertuples(index=True)}

        # dataset sampling
        print(
            f"Sampling dataset to {args.train_samples} train, {args.val_samples} val, {args.test_samples} test samples")
        dataset[train_name] = dataset[train_name].shuffle(seed=42).select(range(args.train_samples))
        dataset[eval_name] = dataset[eval_name].shuffle(seed=42).select(range(args.val_samples))
        dataset[test_name] = dataset[test_name].shuffle(seed=42).select(range(args.test_samples))
        print(
            f"Dataset sampling done, the shapes are {len(dataset[train_name])}, {len(dataset[eval_name])}, {len(dataset[test_name])}")

        # Now add a row_id to each sample
        print("Adding row_id to each sample")
        dataset[train_name] = dataset[train_name].map(lambda example, idx: {'row_id': idx}, with_indices=True)
        dataset[eval_name] = dataset[eval_name].map(lambda example, idx: {'row_id': idx}, with_indices=True)
        dataset[test_name] = dataset[test_name].map(lambda example, idx: {'row_id': idx}, with_indices=True)
        print("Adding row_id done")

        # Now add a row_id to each sample
        dataset[train_name] = dataset[train_name].map(
            lambda example: {'keywords': text_to_keywords(example[question_name])})

        dataset[train_name] = dataset[train_name].map(
            lambda example: {'question': add_special_tokens(example[question_name], example['keywords'])})

        # dataset[train_name] = dataset[train_name].map(
        # lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt'))

        # dataset[train_name] = dataset[train_name].map(
        # lambda example: {'answer_tok': tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})

        dataset[train_name] = dataset[train_name].map(
            lambda example: {
                'graph': text_to_graph_concept(args.graph_depth, example['keywords'], save_dir + '/graphs/',
                                               'train' + str(example['row_id']), nodes_dict, rels_dict)},
            load_from_cache_file=False)

        # dataset[train_name] = dataset[train_name].map(
        #    lambda example: graph_to_nodes_and_rel(example['graph']))

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: {'keywords': text_to_keywords(example[question_name])})

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: {'question': add_special_tokens(example[question_name], example['keywords'])})

        # dataset[eval_name] = dataset[eval_name].map(
        # lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt'))

        # dataset[eval_name] = dataset[eval_name].map(
        # lambda example: {'answer_tok': tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: {
                'graph': text_to_graph_concept(args.graph_depth, example['keywords'], save_dir + '/graphs/',
                                               'val' + str(example['row_id']), nodes_dict, rels_dict)},
            load_from_cache_file=False)

        # dataset[eval_name] = dataset[eval_name].map(
        #    lambda example: graph_to_nodes_and_rel(example['graph']))

        dataset[test_name] = dataset[test_name].map(
            lambda example: {'keywords': text_to_keywords(example[question_name])})

        dataset[test_name] = dataset[test_name].map(
            lambda example: {'question': add_special_tokens(example[question_name], example['keywords'])})
        # dataset[test_name] = dataset[test_name].map(
        # lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt'))

        # dataset[test_name] = dataset[test_name].map(
        # lambda example: {'answer_tok': tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})

        dataset[test_name] = dataset[test_name].map(
            lambda example: {
                'graph': text_to_graph_concept(args.graph_depth, example['keywords'], save_dir + '/graphs/',
                                               'test' + str(example['row_id']), nodes_dict, rels_dict)},
            load_from_cache_file=False)

        # dataset[test_name] = dataset[test_name].map(
        #    lambda example: graph_to_nodes_and_rel(example['graph']))
        print('The number of nodes is:  ', len(nodes))
        print('The number of relations is:  ', len(rels))

        print('dropping empty graphs')
        dataset[train_name] = dataset[train_name].filter(lambda row: '<REL_TOK>' in row['question'])
        dataset[test_name] = dataset[test_name].filter(lambda row: '<REL_TOK>' in row['question'])
        dataset[eval_name] = dataset[eval_name].filter(lambda row: '<REL_TOK>' in row['question'])
        print('Now the lengths of the datasets are as follows:')
        print(len(dataset[train_name]))
        print(len(dataset[test_name]))
        print(len(dataset[eval_name]))

        print('creating nodes and rels embeddings')
        # Load a pretrained model with all-MiniLM-L12-v2 checkpoint
        st_model = SentenceTransformer('all-MiniLM-L12-v2') if device == 'cpu' else SentenceTransformer(
            'all-MiniLM-L12-v2').cuda()
        st_model.max_seq_length = 32

        st_pars = {'convert_to_tensor': True, "batch_size": 256, "show_progress_bar": True}
        nembs = create_memory(st_model, nodes, st_pars)
        rembs = create_memory(st_model, rels, st_pars)
        nembs = Dataset.from_dict(nembs)
        rembs = Dataset.from_dict(rembs)
        pdb.set_trace()
        dataset['memory_rels'] = rembs
        dataset['memory_nodes'] = nembs

        dataset.save_to_disk(save_dir)
    # print(dataset['memory_rels'])

    if not args.skip_train:
        print('dataset loaded')

        setattr(args, 'n_rel', len(dataset['memory_rels'].features))
        setattr(args, 'n_nodes', len(dataset['memory_nodes'].features))
        setattr(args, 'gnn_embs_size', args.sentence_transformer_embedding_size)

        print("In Main")

        # Next I take the date in gg_mm_yyyy format
        date_ = datetime.now().strftime("%d_%m_%Y")
        run_name = date_
        if not args.no_gnn:
            run_name += '_gnn'
        run_name += '_' + str(args.checkpoint_summarizer)
        if args.run_info:
            run_name += '_' + args.run_info

        if not args.no_wandb:
            logger = WandbLogger(
                name=run_name,
                project=args.wandb_project,
            )

        callbacks = []
        early_stopper = EarlyStopping(monitor='val_rouge', patience=args.patience, mode='min')
        callbacks.append(early_stopper)
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
        save_dir = 'checkpoints/' + run_name
        # check the save dir not exists and create it
        if not os.path.exists(save_dir):
            if not args.dont_save:
                os.makedirs(save_dir)
        else:
            print('Save dir already exists, exiting...')
            exit(1)
        if not args.dont_save:
            md_checkpoint = ModelCheckpoint(monitor='val_rouge', save_top_k=args.save_top_k, mode='min',
                                            dirpath=save_dir,
                                            filename='gnnqa-{epoch:02d}-{val_loss:.2f}')
            callbacks.append(md_checkpoint)
        else:
            save_dir = None

        # model creation
        model = T5GNNForConditionalGeneration.from_pretrained(args.checkpoint_summarizer, args)
        gnnqa = GNNQA(model=model, memory_rels=rels,
                      memory_nodes=nodes, tokenizer=tokenizer, save_dir=save_dir,
                      model_lr=args.model_lr, gnn_lr=args.gnn_lr, gnn_layers=args.layer_with_gnn)

        dataset[train_name] = dataset[train_name].map(
            lambda example: {'T5_question': 'question: ' + example['question']})
        dataset[test_name] = dataset[test_name].map(lambda example: {'T5_question': 'question: ' + example['question']})
        dataset[eval_name] = dataset[eval_name].map(lambda example: {'T5_question': 'question: ' + example['question']})

        trainer_args = {
            'max_epochs': args.max_epochs,
            'devices': 1,
            'accumulate_grad_batches': args.accumulate_grad_batches,
            'callbacks': callbacks,
            'enable_checkpointing': not args.dont_save,
            'log_every_n_steps': 1,
            'logger': logger if not args.no_wandb else True,
            'check_val_every_n_epoch': 1,
        }

        trainer = Trainer(**trainer_args)
        trainer.fit(model=gnnqa, train_dataloaders=dataset[train_name], val_dataloaders=dataset[eval_name])

        if args.skip_test:
            return trainer.callback_metrics["val_rouge"].item()  # controllare che ritorni il valore migliore
    if args.skip_test:
        return 0
    results = trainer.test(dataloaders=dataset[test_name], ckpt_path='last' if args.dont_save else 'best')
    print(results)


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
