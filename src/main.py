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
from preprocess import text_to_graph_concept, add_special_tokens, create_memory, \
    graph_to_nodes_and_rel, get_node_and_rel_dict
from data import get_dataset
from model import GNNQA
from t5 import T5GNNForConditionalGeneration, available_reporjection_activations
from pytorch_lightning import Trainer
from sentence_transformers import SentenceTransformer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

torch.manual_seed(42)
seed_everything(42)



def get_args(default=False):
    argparser = argparse.ArgumentParser()


    # RUN Args
    argparser.add_argument('--run_info', type=str, default=None, help='Run info that will be added to run name')
    argparser.add_argument('--wandb_project', type=str, default='gnnqa_default_project', help='wandb project name')
    argparser.add_argument('--debug', action='store_true', help='Debug mode')
    argparser.add_argument('--dont_save', default=False, action='store_true', help='do not save the model')
    argparser.add_argument('--skip_test', default=False, action='store_true', help='skip test')
    argparser.add_argument('--skip_train', default=False, action='store_true', help='skip train')
    argparser.add_argument('--set_anomaly_detection', default=False, action='store_true', help='set torch.autograd.set_detect_anomaly(True) before main')


    # Dataset args
    argparser.add_argument('--dataset', type=str, default='eli5', help='Dataset to use')
    argparser.add_argument('--graph', type=str, default='conceptnet', help='Graph to use')
    argparser.add_argument('--train_samples', type=int, default=None, help='Number of train samples')
    argparser.add_argument('--val_samples', type=int, default=None, help='Number of validation samples')
    argparser.add_argument('--test_samples', type=int, default=None, help='Number of test samples')
    argparser.add_argument('--graph_depth', type=int, default=3, help='Graph depth')
    argparser.add_argument('--keyword_extraction_method', type=str, default='bert', help='kw extraction method')


    # Training args
    argparser.add_argument('--accumulate_grad_batches', type=int, default=8,
                           help='Number of batches to accumulate gradients')
    argparser.add_argument('--load_dataset_from', type=str, default=None, help='Load dataset from path')
    argparser.add_argument('--checkpoint_summarizer', type=str, default='t5-base',
                           help='Summarizer checkpoint from huggingface')
    argparser.add_argument('--max_length', type=int, default=512, help='Max length of the input sequence')
    argparser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    argparser.add_argument('--max_epochs', type=int, default=1, help='max number of epochs')
    argparser.add_argument('--save_top_k', type=int, default=1, help='save top k checkpoints')
    argparser.add_argument('--no_wandb', default=False, action='store_true', help='do not use wandb')
    argparser.add_argument('--optuna_pruner_callback', default=None, help='optuna pruner callback')
    argparser.add_argument('--model_lr', default=0.000001, type=float, help='model learning rate')

    # GNN Args
    argparser.add_argument('--layer_with_gnn', type=int, nargs='+', default=[1, 2], help='Layers with KIL')
    argparser.add_argument('--gnn_topk', type=int, default=2, help='Number of topk nodes to consider for each root node')
    argparser.add_argument('--checkpoint_sentence_transformer', type=str, default='all-MiniLM-L12-v2',
                           help='Load sentence transformer checkpoint')
    argparser.add_argument('--sentence_transformer_embedding_size', type=int, default=384,
                           help='Sentence transformer embedding size')
    argparser.add_argument('--no_gnn', default=False, action='store_true', help='do not use gnn. To lunch baselines')
    argparser.add_argument('--gnn_lr', default=None, type=float, help='gnn learning rate')
    argparser.add_argument('--reprojection_activation', default='tanh', type=str,
                           choices=available_reporjection_activations, help='gnn batch size')

    if not default:
        return argparser.parse_args()
    else:
        return vars(argparser.parse_args([]))

name_mapping = {
    "eli5": ("train_eli5", "validation_eli5", "test_eli5", "title", "answers,text"),
    "conceptnet": ("rel", "arg1", "arg2"),
    "din0s/msmarco-nlgen": ("train", "dev", "test", "query", "answers"),
    "aquamuse": ("train", "validation", "test", "query", "target"),
}


def main(args):

    # take as input the dataset name and return the dataset columns
    dataset_columns = name_mapping.get(args.dataset, None)
    train_name = dataset_columns[0]
    val_name = dataset_columns[1]
    test_name = dataset_columns[2]
    question_name = dataset_columns[3]
    answers_name = dataset_columns[4]

    # set device
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    #pdb.set_trace()

    # load tokenizer and add special tokens
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    tokenizer.add_special_tokens(
        {"additional_special_tokens": tokenizer.additional_special_tokens + ["<REL_TOK>", "<GNN_TOK>"]})

    # load dataset
    if args.load_dataset_from is not None:

        dataset = load_from_disk(args.load_dataset_from)

    else:

        if args.keyword_extraction_method != 'yake' and args.keyword_extraction_method != 'rake':
            args.keyword_extraction_method = 'bert'
        # directory where to save the dataset
        save_dir = f'dataset/{args.dataset}_{args.train_samples}_{args.val_samples}_{args.test_samples}_conceptnet_{args.keyword_extraction_method}'
        # get the original dataset
        dataset = get_dataset(args.dataset)

        if args.dataset == 'din0s/msmarco-nlgen':
            new_set = dataset['train'].train_test_split(test_size=5000)
            dataset[train_name] = new_set['train']
            dataset[test_name] = new_set['test']

        # get all the nodes and relations from the graph and create a dictionary with value and ID (index)
        nodes, rels = get_node_and_rel_dict()
        nodes_dict = {row.custom_value: row.custom_index for row in nodes.itertuples(index=True)}
        rels_dict = {row.custom_value: row.custom_index for row in rels.itertuples(index=True)}

        # dataset sampling
        print(
            f"Sampling dataset to {args.train_samples} train, {args.val_samples} val, {args.test_samples} test samples")
        dataset[train_name] = dataset[train_name].shuffle(seed=42).select(range(args.train_samples))
        dataset[val_name] = dataset[val_name].shuffle(seed=42).select(range(args.val_samples))
        dataset[test_name] = dataset[test_name].shuffle(seed=42).select(range(args.test_samples))
        print(
            f"Dataset sampling done, the shapes are {len(dataset[train_name])}, {len(dataset[val_name])}, {len(dataset[test_name])}")

        # Now add a row_id to each sample
        print("Adding row_id to each sample")
        dataset[train_name] = dataset[train_name].map(lambda example, idx: {'row_id': idx}, with_indices=True)
        dataset[val_name] = dataset[val_name].map(lambda example, idx: {'row_id': idx}, with_indices=True)
        dataset[test_name] = dataset[test_name].map(lambda example, idx: {'row_id': idx}, with_indices=True)
        print("Adding row_id done")

        # Now add the rows: keywords, question, graph
        #dataset[train_name] = dataset[train_name].map(
            #lambda example: {'keywords': text_to_keywords(example[question_name])})

        dataset[train_name] = dataset[train_name].map(
            lambda example: text_to_graph_concept(args.graph_depth, example[question_name], save_dir + '/graphs/',
                                               'train' + str(example['row_id']), nodes_dict, rels_dict, args),
            load_from_cache_file=False)

        dataset[train_name] = dataset[train_name].map(
            lambda example: {'question': add_special_tokens(example[question_name], example['keywords'])})

        # dataset[train_name] = dataset[train_name].map(
        # lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt'))

        # dataset[train_name] = dataset[train_name].map(
        # lambda example: {'answer_tok': tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})



        # dataset[train_name] = dataset[train_name].map(
        #    lambda example: graph_to_nodes_and_rel(example['graph']))

        #dataset[val_name] = dataset[val_name].map(
            #lambda example: {'keywords': text_to_keywords(example[question_name])})

        dataset[val_name] = dataset[val_name].map(
            lambda example:text_to_graph_concept(args.graph_depth, example[question_name], save_dir + '/graphs/',
                                                 'val' + str(example['row_id']), nodes_dict, rels_dict, args),
            load_from_cache_file=False)

        dataset[val_name] = dataset[val_name].map(
            lambda example: {'question': add_special_tokens(example[question_name], example['keywords'])})

        # dataset[val_name] = dataset[val_name].map(
        # lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt'))

        # dataset[val_name] = dataset[val_name].map(
        # lambda example: {'answer_tok': tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})



        # dataset[val_name] = dataset[val_name].map(
        #    lambda example: graph_to_nodes_and_rel(example['graph']))

        #dataset[test_name] = dataset[test_name].map(
            #lambda example: {'keywords': text_to_keywords(example[question_name])})

        dataset[test_name] = dataset[test_name].map(
            lambda example: text_to_graph_concept(args.graph_depth, example[question_name], save_dir + '/graphs/',
                                               'test' + str(example['row_id']), nodes_dict, rels_dict, args),
            load_from_cache_file=False)

        dataset[test_name] = dataset[test_name].map(
            lambda example: {'question': add_special_tokens(example[question_name], example['keywords'])})
        # dataset[test_name] = dataset[test_name].map(
        # lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt'))

        # dataset[test_name] = dataset[test_name].map(
        # lambda example: {'answer_tok': tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})



        # dataset[test_name] = dataset[test_name].map(
        #    lambda example: graph_to_nodes_and_rel(example['graph']))

        print('The number of nodes is:  ', len(nodes))
        print('The number of relations is:  ', len(rels))

        print('dropping empty graphs')
        dataset[train_name] = dataset[train_name].filter(lambda row: '<REL_TOK>' in row['question'])
        dataset[val_name] = dataset[val_name].filter(lambda row: '<REL_TOK>' in row['question'])
        dataset[test_name] = dataset[test_name].filter(lambda row: '<REL_TOK>' in row['question'])
        print('Now the lengths of the datasets are as follows:')
        print(len(dataset[train_name]))
        print(len(dataset[val_name]))
        print(len(dataset[test_name]))

        print('creating nodes and rels embeddings')
        # Load a pretrained model with all-MiniLM-L12-v2 checkpoint
        st_model = SentenceTransformer('all-MiniLM-L12-v2') if device == 'cpu' else SentenceTransformer(
            'all-MiniLM-L12-v2').cuda()
        st_model.max_seq_length = 32
        st_pars = {'convert_to_tensor': True, "batch_size": 256, "show_progress_bar": True}
        # use st to all the nodes and rels of the graphs and save it to the dataset
        nembs = create_memory(st_model, nodes, st_pars)
        rembs = create_memory(st_model, rels, st_pars)
        dataset['memory_nodes'] = Dataset.from_pandas(pd.DataFrame(data=nembs))
        dataset['memory_rels'] = Dataset.from_pandas(pd.DataFrame(data=rembs))

        # save the dataset to disk
        dataset.save_to_disk(save_dir)

    # print(dataset['memory_rels'])

    print('dataset loaded')

    if not args.skip_train:

        print("In Main")

        if args.train_samples:
            dataset[train_name] = dataset[train_name].select(range(args.train_samples))
        if args.val_samples:
            dataset[val_name] = dataset[val_name].select(range(args.val_samples))
        if args.test_samples:
            dataset[test_name] = dataset[test_name].select(range(args.test_samples))

        # set total number of rel, nodes and gnn embs size
        setattr(args, 'n_rel', len(dataset['memory_rels'].features))
        setattr(args, 'n_nodes', len(dataset['memory_nodes'].features))
        setattr(args, 'gnn_embs_size', args.sentence_transformer_embedding_size)





        # Next I take the date in gg_mm_yyyy format
        date_ = datetime.now().strftime("%d_%m_%Y")
        run_name = date_
        if not args.no_gnn:
            run_name += '_gnn'
        run_name += '_' + str(args.checkpoint_summarizer)
        if args.run_info:
            run_name += '_' + args.run_info

        # set wandb logger
        if not args.no_wandb:
            logger = WandbLogger(
                name=run_name,
                project=args.wandb_project,
            )
        else:
            logger = TensorBoardLogger(
                save_dir='logs/',
                name=run_name,
            )

        # set callbacks
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
            if not args.dont_save:
                print('Save dir already exists, exiting...')
                exit(1)
        if not args.dont_save:
            md_checkpoint = ModelCheckpoint(monitor='val_rouge', save_top_k=args.save_top_k, mode='min',
                                            dirpath=save_dir,
                                            filename='gnnqa-{epoch:02d}-{val_loss:.2f}')
            callbacks.append(md_checkpoint)
        else:
            save_dir = None

        # create dict with ID and word for each nodes and rels
        nodes = {i: word for i, word in enumerate(dataset['memory_nodes'].features)}
        rels = {i: word for i, word in enumerate(dataset['memory_rels'].features)}

        # model creation
        model = T5GNNForConditionalGeneration.from_pretrained(args.checkpoint_summarizer, args)
        gnnqa = GNNQA(model=model, ids_to_rels=rels, ids_to_nodes=nodes,
                      memory_embs=dataset['memory_nodes'].to_dict(), tokenizer=tokenizer, save_dir=save_dir,
                      model_lr=args.model_lr, gnn_lr=args.gnn_lr, gnn_layers=args.layer_with_gnn, labels=answers_name)

        # create T5 question for each example
        dataset[train_name] = dataset[train_name].map(
            lambda example: {'T5_question': 'question: ' + example['question']})
        dataset[val_name] = dataset[val_name].map(lambda example: {'T5_question': 'question: ' + example['question']})
        dataset[test_name] = dataset[test_name].map(lambda example: {'T5_question': 'question: ' + example['question']})

        trainer_args = {
            'max_epochs': args.max_epochs,
            'devices': 1,
            'accumulate_grad_batches': args.accumulate_grad_batches,
            'callbacks': callbacks,
            'enable_checkpointing': not args.dont_save,
            'log_every_n_steps': 1,
            'logger': logger,
            'check_val_every_n_epoch': 1,
            'deterministic': True,
        }

        trainer = Trainer(**trainer_args)
        trainer.fit(model=gnnqa, train_dataloaders=dataset[train_name], val_dataloaders=dataset[val_name])

        if args.skip_test:
            return trainer.callback_metrics["val_rouge"].item()  # controllare che ritorni il valore migliore
    if args.skip_test:
        return 0
    results = trainer.test(dataloaders=dataset[test_name], ckpt_path='last' if args.dont_save else 'best')
    print(results)


if __name__ == '__main__':
    args = get_args()
    if args.set_anomaly_detection:
        with torch.autograd.set_detect_anomaly(True):
            main(args)
    else:
        main(args)
