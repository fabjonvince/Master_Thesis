import pdb
import re
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import yake
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake
nltk.download('stopwords')
nltk.download('punkt')

import numpy as np
import os

import pandas as pd
from keybert import KeyBERT

from data import get_dataset

"""
def text_to_keywords(
        text,  # domanda
):
    kw_model = KeyBERT()
    kw = kw_model.extract_keywords(text)
    txt = [kw[i][0].lower() for i in range(len(kw))]
    txt = np.unique(txt)

    return txt
"""


def add_special_tokens(
        question,  # domanda
        kw,  # keywords
):
    new_question = question
    for word in kw:
        idx = re.search(r"\b({})\b".format(word), new_question, re.IGNORECASE)
        if idx is not None:
            idx = idx.start()
            new_question = new_question[:idx] + "<REL_TOK><GNN_TOK>" + new_question[idx:]


    return new_question


def get_node_and_rel_dict():
    graph = get_dataset('conceptnet')
    graph = graph['train']
    graph = graph.to_pandas()
    nodes = list(graph['arg1'].unique())
    rels = list(graph['rel'].unique())
    rels.append('self')
    nodes.extend(list(graph['arg2'].unique()))
    nodes = list(set(nodes))
    node_index = np.arange(len(nodes), dtype=np.uint32)
    rel_index = np.arange(len(rels), dtype=np.uint32)
    rels_dict = [{'custom_index': i, 'custom_value': v} for i, v in zip(rel_index, rels)]
    nodes_dict = [{'custom_index': i, 'custom_value': v} for i, v in zip(node_index, nodes)]

    return pd.DataFrame(data=nodes_dict), pd.DataFrame(data=rels_dict)




def serialize(triplets):
    div = '<[^_^]>'
    ser = [s + div + r + div + e for s, r, e in triplets]
    return ser


def deserialize(texts):
    div = '<[^_^]>'
    deser = [(s, r, e) for text in texts for s, r, e in text.split(div)]
    return deser


def save_with_pickle(dir_to_save, triplets):
    with open(dir_to_save, 'w') as fp:
        pickle.dump(triplets, fp)


def load_with_pickle(dir_to_load):
    with open(dir_to_load, "r") as fp:
        data = pickle.load(fp)
    return data

def from_triplets_of_ids_to_triplets_of_string(triplets, node_dict, rel_dict):
    str_trips = [(node_dict[s], rel_dict[r], node_dict[e]) for s,r,e in triplets]
    return str_trips


def text_to_graph_concept(
        N,  # numero di salti
        text,  # domanda
        save_dir,
        row_id,
        nodes_dict,
        rels_dict,
        args,
):

    # here we take all the words as keywords except the stopwords
    if args.keyword_extraction_method == 'all':
        stop_words = set(stopwords.words('english'))
        word_tok = word_tokenize(text)
        kw = [w.lower() for w in word_tok if w.lower() not in stop_words]
    # look which keywords extraction method to use and apply it
    elif args.keyword_extraction_method == 'rake':
        kw_model = Rake()
        kw_model.extract_keywords_from_text(text)
        kw = kw_model.get_ranked_phrases()
        kw = [kw[i].lower() for i in range(len(kw))]
    else:
        if args.keyword_extraction_method == 'yake':
            kw_model = yake.KeywordExtractor()
        else:
            kw_model = KeyBERT()
        kw = kw_model.extract_keywords(text)
        kw = [kw[i][0].lower() for i in range(len(kw))]

    kw = np.unique(kw)
    txt = kw

    obj = 'arg2'
    graph = get_dataset('conceptnet')
    graph = graph['train']
    graph = graph.to_pandas()
    graph.set_index('arg1', inplace=True)
    triplets_list = []
    entities_list = kw

    for i in range(N):
        kw = [k for k in (set(kw) & set(graph.index))]
        if i == 0:
            txt = kw

        filtered = graph.loc[kw]
        triplets = [(nodes_dict[row.Index], rels_dict[row.rel], nodes_dict[row.arg2]) for row in filtered.itertuples(index=True)]
        triplets = np.unique(triplets, axis=0)
        triplets_list.extend([triplet for triplet in triplets])

        kw = filtered.loc[~filtered[obj].isin(entities_list)][obj].drop_duplicates().to_numpy()

        entities_list = np.hstack((entities_list, kw))
    triplets_list.extend([(nodes_dict[entity], rels_dict['self'], nodes_dict[entity]) for entity in entities_list if entity in nodes_dict])

    # Now save the triplets_list into the directory save_dir with the name row_id.graph
    # check if save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # check if save_dir ends with '/'
    if save_dir[-1] != '/':
        save_dir += '/'
    np.save(save_dir + str(row_id) + '.graph.npy', triplets_list)
    return {'keywords': txt, 'graph': save_dir + str(row_id) + '.graph.npy'}


def print_triplets(triplets):
    for triplet in triplets:
        print(triplet[0] + " -> " + triplet[1] + " -> " + triplet[2])

"""
def graph_to_nodes_and_rel(triplets):
    edges = {}
    relations = {}

    for rel in triplets:

        if rel[0] in edges.keys():
            if rel[1] not in edges[rel[0]]:
                edges[rel[0]].append(rel[1])
        else:
            edges[rel[0]] = [rel[1]]

        if rel[2] in edges.keys():
            if rel[1] not in edges[rel[2]]:
                edges[rel[2]].append(rel[1])
        else:
            edges[rel[2]] = [rel[1]]

        if rel[1] in relations.keys():
            relations[rel[1]].append((rel[0], rel[2]))
        else:
            relations[rel[1]] = [(rel[0], rel[2])]

    return {'nodes': edges, 'relations': relations}
"""

def extract_support_from_links(support, dataset):
    txt = ''
    if dataset == 'din0s':
        txt = ' '.join(x['passage_text'] for x in support if x['is_selected'] == 1)
    elif dataset == 'aquamuse':
        if len(support) > 10:
            support = support[:10]
        for url in support:
            try:
                html = requests.get(url, timeout=2)
                soup = BeautifulSoup(html.text, 'html.parser')
                txt = txt + ' ' + soup.getText()
            except requests.exceptions.RequestException as e:
                pass

        txt = txt.replace('\n', ' ')

    return txt



def create_memory(model, sentences, args):
    embs = model.encode(sentences['custom_value'], **args).to('cpu')
    embs = {s:e for s,e in zip(sentences['custom_value'], embs)}
    return embs


def create_embeddings_with_model(model, nodes, tokenizer, batch_size, dir, device):

    memory_embs = {}
    # tokenize all the nodes
    node_tok = tokenizer(nodes, padding='max_length', truncation=True, max_length=32,
                         return_tensors='pt')['input_ids'].to('cpu')

    # create batch of the nodes, embed them, compute mean and save them
    for i in range(0, 800, batch_size):
        selected_nodes = node_tok[i: i + batch_size]
        embedded = model.shared(selected_nodes)
        for j, node in enumerate(nodes[i: i + batch_size]):
            memory_embs[node] = torch.mean(embedded[j], dim=0).detach().cpu()
        np.save(f'{dir}/embed_{i}_{i + batch_size}.npy', memory_embs)


