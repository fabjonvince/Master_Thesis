import argparse
import time
from os.path import exists
import pdb

import numpy as np
from datasets import load_from_disk
from transformers import T5Tokenizer

from preprocess import text_to_graph, graph_to_rel, graph_to_edges, rel_to_adj, print_info_triples
from data import get_dataset
from model import GNNQA, T5DataModule
from t5 import T5KILForConditionalGeneration
from pytorch_lightning import Trainer
from sentence_transformers import SentenceTransformer

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='conceptnet', help='Dataset to use')
argparser.add_argument('--train_samples', type=int, default=10, help='Number of train samples')
argparser.add_argument('--val_samples', type=int, default=10, help='Number of validation samples')
argparser.add_argument('--test_samples', type=int, default=10, help='Number of test samples')
argparser.add_argument('--layer_with_kil', type=int, default=[1, 2], help='Layers with KIL')
argparser.add_argument('--debug', action='store_true', help='Debug mode')
name_mapping = {
"eli5": ("train_eli5", "validation_eli5", "test_eli5", "title", "answers"),
"conceptnet": ("rel", "arg1", "arg2"),
}


def raw_input(param):
    pass


def main(args):
    pdb.set_trace()
    print("In Main")

    dataset_columns = name_mapping.get(args.dataset, None)

    if args.dataset == 'eli5':
        train_name = dataset_columns[0]
        eval_name = dataset_columns[1]
        test_name = dataset_columns[2]
        question_name = dataset_columns[3]
        answers_name = dataset_columns[4]

        tokenizer = T5Tokenizer.from_pretrained('t5-base')

        #sistemare test set (non ridotto) e usato il map del validation invece che del test

        #load dataset
        #if exists('dataset/eli5_10'):

            #dataset = load_from_disk('dataset/eli5_10')

        #else:

        dataset = get_dataset(args.dataset)

        # dataset sampling
        dataset[train_name] = dataset[train_name].shuffle(seed=42).select(range(args.train_samples))
        dataset[eval_name] = dataset[eval_name].shuffle(seed=42).select(range(args.val_samples))
        dataset[test_name] = dataset[test_name].shuffle(seed=42).select(range(args.test_samples))

        dataset[train_name] = dataset[train_name].map(
            lambda example: tokenizer(example[answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'))

        dataset[train_name] = dataset[train_name].map(
            lambda example: {'answer_tok': tokenizer(example[question_name], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})

        #dataset[train_name] = dataset[train_name].map(lambda example: {'graph': text_to_graph(2, example[question_name])})

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

        print(dataset[train_name][0]['graph'])

        #print_info_triples(dataset[train_name][0]['graph'])

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

        # print(dataset[train_name][0]['relations'])

    elif args.dataset == 'conceptnet':
        rel = dataset_columns[0]
        subj = dataset_columns[1]
        obj = dataset_columns[2]

        checkpoint = input("Insert checkpoint to use: ") #insert checkpoint for sentence transformer, separated by space
        checkpoint = checkpoint.split(' ')

        dataset = get_dataset(args.dataset)
        dataset = dataset['train']

        list_trip = dataset.to_list()

        triplets = [(item[subj], item[rel], item[obj]) for item in list_trip]

        nodes = graph_to_edges(triplets)
        sub_keys = list(nodes.keys())[:100]

        enc_nod_check = {}
        tempi = []

        for mc in checkpoint:
            start_time = time.time()
            model_sent = SentenceTransformer(mc)
            model_sent.max_seq_length = 12
            enc_nod_check[mc] = [{key: model_sent.encode(nodes[key], batch_size=32)} for key in sub_keys]
            tempi.append(mc + " " + str(time.time() - start_time))

        for mc in checkpoint:
            start_time = time.time()
            model_sent = SentenceTransformer(mc)
            model_sent.max_seq_length = 12
            enc_nod_check[mc] = [{key: model_sent.encode(nodes[key], batch_size=128)} for key in sub_keys]
            tempi.append(mc + " " + str(time.time() - start_time))

        for mc in checkpoint:
            start_time = time.time()
            model_sent = SentenceTransformer(mc)
            model_sent.max_seq_length = 12
            enc_nod_check[mc] = [{key: model_sent.encode(nodes[key], batch_size=256)} for key in sub_keys]
            tempi.append(mc + " " + str(time.time() - start_time))

        print(tempi)

        print(nodes)

        """
        'paraphrase-MiniLM-L3-v2 47.77237868309021', 
        'paraphrase-albert-small-v2 55.20635676383972', 
        'all-MiniLM-L6-v2 69.34488487243652', 
        
        'paraphrase-MiniLM-L3-v2 49.446166038513184',
        'paraphrase-albert-small-v2 54.45325207710266',
        'all-MiniLM-L6-v2 67.73390364646912',
        
        'paraphrase-MiniLM-L3-v2 49.50322461128235', 
        'paraphrase-albert-small-v2 54.93287134170532',
        'all-MiniLM-L6-v2 66.79923272132874'
        
        #  paraphrase-MiniLM-L3-v2 paraphrase-albert-small-v2 all-MiniLM-L6-v2
        #     speed      smallest size   faste and good quality
        """

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


