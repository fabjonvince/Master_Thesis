import pdb
import re
import time

import numpy as np
from keybert import KeyBERT

from data import get_dataset


def text_to_keywords(
        text, #domanda
        ):

    kw_model = KeyBERT()
    kw = kw_model.extract_keywords(text)
    txt = [kw[i][0].lower() for i in range(len(kw))]
    txt = np.unique(txt)

    return txt



def add_special_tokens(
        question, #domanda
        kw, #keywords
        ):

    new_question = question
    for word in kw:
        idx = re.search(r"\b({})\b".format(word), new_question, re.IGNORECASE).start()
        new_question = new_question[:idx] + "<REL_TOKEN><GNN_TOK> " + new_question[idx:]

    return new_question



def text_to_graph_concept(
        N, #numero di salti
        kw, #domanda
        ):
    obj = 'arg2'

    graph = get_dataset('conceptnet')
    graph = graph['train']
    graph = graph.to_pandas()
    graph.set_index('arg1', inplace=True)
    triplets_list = []
    entities_list = kw

    for i in range(N):
        kw = [k for k in (set(kw) & set(graph.index))]
        #filtered = graph.loc[kw, [subj, rel, obj]] #
        filtered = graph.loc[kw]
        #triplets = filtered.to_numpy()
        #triplets = [(item[0], item[1], item[2]) for item in triplets] #
        triplets = [(row.Index, row.rel, row.arg2) for row in filtered.itertuples(index=True)]
        triplets = np.unique(triplets, axis=0)
        triplets_list.extend([triplet for triplet in triplets])

        kw = filtered.loc[~filtered[obj].isin(entities_list)][obj].drop_duplicates().to_numpy()

        entities_list = np.hstack((entities_list, kw))

    triplets_list.extend([(entity, 'self', entity) for entity in entities_list])

    return triplets_list


def print_triplets(triplets):
    for triplet in triplets:
        print(triplet[0] + " -> " + triplet[1] + " -> " + triplet[2])


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


def create_memory(model, sentences, args):
    embs = model.encode(sentences, **args)
    embeddings = {k: v for k, v in zip(sentences, embs)}

    return embeddings






