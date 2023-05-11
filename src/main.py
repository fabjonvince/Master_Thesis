import argparse
from os.path import exists
import pdb

import numpy as np
from datasets import load_from_disk
from transformers import T5Tokenizer

from preprocess import text_to_graph, print_triplets, graph_to_rel, graph_to_edges, rel_to_adj
from data import get_dataset
from model import GNNQA, T5DataModule
from t5 import T5KILForConditionalGeneration
from pytorch_lightning import Trainer
from sentence_transformers import SentenceTransformer

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='eli5', help='Dataset to use')
argparser.add_argument('--train_samples', type=int, default=10, help='Number of train samples')
argparser.add_argument('--layer_with_kil', type=int, default=[1, 2], help='Layers with KIL')
name_mapping = {
"eli5": ("train_eli5", "validation_eli5", "test_eli5", "title", "answers")
}


def main(args):
    pdb.set_trace()
    print("In Main")

    dataset_columns = name_mapping.get(args.dataset, None)
    train_name = dataset_columns[0]
    eval_name = dataset_columns[1]
    test_name = dataset_columns[2]
    question_name = dataset_columns[3]
    answers_name = dataset_columns[4]

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    #sistemare test set (non ridotto) e usato il map del validation invece che del test

    #load dataset
    if exists('dataset/eli5_10'):

        dataset = load_from_disk('dataset/eli5_10')

    else:

        dataset = get_dataset(args.dataset)

        # dataset sampling
        dataset[train_name] = dataset[train_name].shuffle(seed=42).select(range(args.train_samples))
        dataset[eval_name] = dataset[eval_name].shuffle(seed=42).select(range(args.train_samples))
        dataset[test_name] = dataset[test_name].shuffle(seed=42).select(range(args.train_samples))

        dataset[train_name] = dataset[train_name].map(
            lambda example: tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'))

        dataset[train_name] = dataset[train_name].map(
            lambda example: {'answer_tok': tokenizer(example[question_name], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})

        dataset[train_name] = dataset[train_name].map(lambda example: {'graph': text_to_graph(2, example[question_name])})

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'))

        dataset[eval_name] = dataset[eval_name].map(lambda example: {
            'answer_tok': tokenizer(example[question_name], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})

        dataset[eval_name] = dataset[eval_name].map(
            lambda example: {'graph': text_to_graph(2, example[question_name])})

        dataset[test_name] = dataset[test_name].map(
            lambda example: tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'))

        dataset[test_name] = dataset[test_name].map(lambda example: {
            'answer_tok': tokenizer(example[question_name], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})

        dataset[test_name] = dataset[test_name].map(
            lambda example: {'graph': text_to_graph(2, example[question_name])})

        dataset.save_to_disk('dataset/eli5_10')


    print(len(dataset[train_name]))

    # prova con una singola frase
    #triplets = text_to_graph(2, dataset[train_name][0][question_name])

    # dalle triple ottenute crare una lista di nodi e relazioni
    dataset[train_name] = dataset[train_name].map(lambda example: {'relations' : graph_to_rel(example['graph'])})
    dataset[train_name] = dataset[train_name].map(lambda example: {'edges' : graph_to_edges(example['graph'])})
    dataset[train_name] = dataset[train_name].map(lambda example: {'adj': rel_to_adj(example['relations'])})

    """
    dataset[eval_name] = dataset[eval_name].map(lambda example: {'relations': graph_to_rel(example['graph'])})
    dataset[eval_name] = dataset[eval_name].map(lambda example: {'edges': graph_to_edges(example['graph'])})
    dataset[eval_name] = dataset[eval_name].map(lambda example: {'adj': rel_to_adj(example['relations'])})

    dataset[test_name] = dataset[test_name].map(lambda example: {'relations': graph_to_rel(example['graph'])})
    dataset[test_name] = dataset[test_name].map(lambda example: {'edges': graph_to_edges(example['graph'])})
    dataset[test_name] = dataset[test_name].map(lambda example: {'adj': rel_to_adj(example['relations'])})
    """

    #print(dataset[train_name][0]['relations'])

    nrel = 5 # to do: count number of relations
    setattr(args, 'nrel', nrel)

    nnodes = 5 #da modificare
    setattr(args, 'nnodes', nnodes)

    #model creation
    model = T5KILForConditionalGeneration.from_pretrained('t5-base', args)
    transformer_rel = SentenceTransformer('all-mpnet-base-v2')
    transformer_nodes = SentenceTransformer('all-mpnet-base-v2')
    gnnqa = GNNQA(model=model, rel_model=transformer_rel, nodes_model=transformer_nodes)
    trainer_args = {'max_epochs': 1, 'gpus': 1}

    trainer = Trainer()
    trainer.fit(model=gnnqa, train_dataloaders=dataset[train_name])#, val_dataloaders=dataset[eval_name])

    trainer.train()

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)


