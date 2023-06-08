import argparse
import os
import time
import pdb

import numpy as np
from datasets import load_from_disk
from transformers import T5Tokenizer, TrainingArguments
import wandb
from preprocess import text_to_graph_concept, add_special_tokens, text_to_keywords, create_memory, graph_to_nodes_and_rel
from data import get_dataset
from model import GNNQA
from t5 import T5GNNForConditionalGeneration
from pytorch_lightning import Trainer
from sentence_transformers import SentenceTransformer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='eli5', help='Dataset to use')
argparser.add_argument('--graph', type=str, default='conceptnet', help='Graph to use')
argparser.add_argument('--train_samples', type=int, default=10, help='Number of train samples')
argparser.add_argument('--val_samples', type=int, default=10, help='Number of validation samples')
argparser.add_argument('--test_samples', type=int, default=10, help='Number of test samples')
argparser.add_argument('--layer_with_gnn', type=int, default=[1, 2], help='Layers with KIL')
argparser.add_argument('--debug', action='store_true', help='Debug mode')
argparser.add_argument('--gnn_topk', type=int, default=2, help='Number of topk nodes to consider for each root node')
argparser.add_argument('--load_dataset_from', type=str, default=None, help='Load dataset from path')
argparser.add_argument('--checkpoint_sentence_transformer', type=str, default='all-MiniLM-L12-v2', help='Load sentence transformer checkpoint')
argparser.add_argument('--sentence_transformer_embedding_size', type=int, default=384, help='Sentence transformer embedding size')
argparser.add_argument('--max_length', type=int, default=512, help='Max length of the input sequence')
argparser.add_argument('--graph_depth', type=int, default=3, help='Graph depth')
argparser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
argparser.add_argument('--gpus', type=int, default='gpu', help='Gpus')
argparser.add_argument('--max_epochs', type=int, default=1, help='max number of epochs')
argparser.add_argument('--save_top_k', type=int, default=1, help='save top k checkpoints')
argparser.add_argument('--optuna_pruner_callback', type=str, default=None, help='optuna pruner callback')
argparser.add_argument('--skip_test', default=False, action='store_true', help='skip test')
name_mapping = {
"eli5": ("train_eli5", "validation_eli5", "test_eli5", "title", "answers"),
"conceptnet": ("rel", "arg1", "arg2"),
}


def main(args):
    pdb.set_trace()
    print("In Main")
    wandb.login(key='342bec801c9a0a847e9479c10f2b1dd5e3f8261b')

    dataset_columns = name_mapping.get(args.dataset, None)
    train_name = dataset_columns[0]
    eval_name = dataset_columns[1]
    test_name = dataset_columns[2]
    question_name = dataset_columns[3]
    answers_name = dataset_columns[4]

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    #load dataset
    if args.load_dataset_from is not None:

        dataset = load_from_disk('dataset/eli5_10_conceptnet')

    else:

        dataset = get_dataset(args.dataset)

        # dataset sampling
        dataset[train_name] = dataset[train_name].shuffle(seed=42).select(range(args.train_samples))
        dataset[eval_name] = dataset[eval_name].shuffle(seed=42).select(range(args.val_samples))
        dataset[test_name] = dataset[test_name].shuffle(seed=42).select(range(args.test_samples))

        tokenizer.add_special_tokens({'additional_special_tokens': ['<REL_TOK>', '<GNN_TOK>']})


        dataset[train_name] = dataset[train_name].map(
            lambda example: {'keywords': text_to_keywords(example[question_name])})

        dataset[train_name] = dataset[train_name].map(
            lambda example: {'question': add_special_tokens(example[question_name], example['keywords'])})

        dataset[train_name] = dataset[train_name].map(
            lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt'))

        dataset[train_name] = dataset[train_name].map(
            lambda example: {'answer_tok': tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt')})

        dataset[train_name] = dataset[train_name].map(
            lambda example: {'graph': text_to_graph_concept(args.graph_depth, example['keywords'])})

        dataset[train_name] = dataset[train_name].map(
            lambda example: graph_to_nodes_and_rel(example['graph']))




        dataset[eval_name] = dataset[eval_name].map(
            lambda example: {'keywords': text_to_keywords(example[question_name])})

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: {'question': add_special_tokens(example[question_name], example['keywords'])})

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt'))

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: {'answer_tok': tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt')})

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: {'graph': text_to_graph_concept(args.graph_depth, example['keywords'])})

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: graph_to_nodes_and_rel(example['graph']))




        dataset[test_name] = dataset[test_name].map(
            lambda example: {'keywords': text_to_keywords(example[question_name])})

        dataset[test_name] = dataset[test_name].map(
            lambda example: {'question': add_special_tokens(example[question_name], example['keywords'])})

        dataset[test_name] = dataset[test_name].map(
            lambda example: tokenizer(example['question'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt'))

        dataset[test_name] = dataset[test_name].map(
            lambda example: {'answer_tok': tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=args.max_length, return_tensors='pt')})

        dataset[test_name] = dataset[test_name].map(
            lambda example: {'graph': text_to_graph_concept(args.graph_depth, example['keywords'])})

        dataset[test_name] = dataset[test_name].map(
            lambda example: graph_to_nodes_and_rel(example['graph']))


        dataset.save_to_disk('dataset/eli5_10_conceptnet')


    nodes = np.unique([s for d in dataset[train_name]['graph'] for s,_,s in d] + \
                     [s for d in dataset[eval_name]['graph'] for s,_,s in d] + \
                     [s for d in dataset[test_name]['graph'] for s,_,s in d])

    rels = np.unique([k for d in dataset[train_name]['graph'] for _,k,_ in d] + \
                     [k for d in dataset[eval_name]['graph'] for _,k,_ in d] + \
                     [k for d in dataset[test_name]['graph'] for _,k,_ in d])

    # Load a pretrained model with all-MiniLM-L12-v2 checkpoint
    model = SentenceTransformer('all-MiniLM-L12-v2')
    memory_nodes = create_memory(model, nodes, {'convert_to_tensor': True})
    memory_rels = create_memory(model, rels, {'convert_to_tensor': True})

    setattr(args, 'n_rel', len(memory_rels))
    setattr(args, 'n_nodes', len(memory_nodes))
    setattr(args, 'gnn_embs_size', args.sentence_transformer_embedding_size)

    """
    run = wandb.init(project='tesim', config={
        'epochs': args.max_epochs,
    })
    """
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = "tesim"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    # model creation
    model = T5GNNForConditionalGeneration.from_pretrained('t5-base', args)
    gnnqa = GNNQA(model=model, memory_rels=memory_rels, memory_nodes=memory_nodes)
    #training_args = TrainingArguments(report_to='wandb')
    trainer_args = {'accelerator' : args.gpus, 'max_epochs' : args.max_epochs}

    early_stopper = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')
    md_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=args.save_top_k, mode='min', dirpath='checkpoints', filename='gnnqa-{epoch:02d}-{val_loss:.2f}')
    if args.optuna_pruner_callback is None:
        trainer = Trainer(trainer_args, callbacks=[early_stopper, md_checkpoint])
        #trainer = Trainer(trainer_args, args=training_args, callbacks=[early_stopper, md_checkpoint])
        #else aggiungere optuna pruner alle callbacks

    trainer.fit(model=gnnqa, train_dataloaders=dataset[train_name], val_dataloaders=dataset[eval_name])

    wandb.log({'val_loss': trainer.callback_metrics['val_loss'], 'accuracy': trainer.callback_metrics['accuracy']})

    results = trainer.predict(model=gnnqa, test_dataloaders=dataset[test_name])

    print(results)



if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)


