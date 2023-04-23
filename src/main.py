import argparse
from os.path import exists
import pdb

import numpy as np
from datasets import load_from_disk
from transformers import T5Tokenizer

from preprocess import text_to_graph, print_triplets, graph_to_rel, graph_to_nodes
from data import get_dataset
from model import GNNQA, T5DataModule
from t5 import T5KILForConditionalGeneration
from pytorch_lightning import Trainer

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='eli5', help='Dataset to use')
argparser.add_argument('--train_samples', type=int, default=1000, help='Number of train samples')
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

    #load dataset
    if exists('dataset/eli5_1000'):

        dataset = load_from_disk('dataset/eli5_1000')

    else:

        dataset = get_dataset(args.dataset)

        # dataset sampling
        dataset[train_name] = dataset[train_name].shuffle(seed=42).select(range(args.train_samples))
        dataset[eval_name] = dataset[eval_name].shuffle(seed=42).select(range(args.train_samples))

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

        dataset[test_name] = dataset[eval_name].map(
            lambda example: tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'))

        dataset[test_name] = dataset[eval_name].map(lambda example: {
            'answer_tok': tokenizer(example[question_name], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})

        dataset[test_name] = dataset[eval_name].map(
            lambda example: {'graph': text_to_graph(2, example[question_name])})

        dataset.save_to_disk('dataset/eli5_1000')


    print(len(dataset[train_name]))

    # prova con una singola frase
    #triplets = text_to_graph(2, dataset[train_name][0][question_name])


    # dalle triple ottenute crare una lista di nodi
    #dataset[train_name] = dataset[train_name].map(lambda example: {'nodes': graph_to_nodes(example['graph'])})
    #dataset[eval_name] = dataset[eval_name].map(lambda example: {'nodes': graph_to_nodes(example['graph'])})

    # dalle triple ottenute crare un dizionario con le relazioni per ogni nodo
    #dataset[train_name] = dataset[train_name].map(lambda example: {'rel': graph_to_rel(example['graph'])})
    #dataset[eval_name] = dataset[eval_name].map(lambda example: {'rel': graph_to_rel(example['graph'])})


    print(dataset[train_name][0]['graph'])


    nrel = 5 # to do: count number of relations
    setattr(args, 'nrel', nrel)

    nnodes = len(np.unique(np.concatenate([item for triple in dataset[train_name]['graph'] for item in triple]))) #da modificare
    setattr(args, 'nnodes', nnodes)

    #model creation
    model = T5KILForConditionalGeneration.from_pretrained('t5-base', args)
    gnnqa = GNNQA(model)
    trainer_args = {'max_epochs': 1, 'gpus': 1}

    trainer = Trainer()
    trainer.fit(model=gnnqa, train_dataloaders=dataset[train_name])#, val_dataloaders=dataset[eval_name])

    trainer.train()

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)


