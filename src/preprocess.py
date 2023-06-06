import pdb
import re
import time

import numpy as np
import pandas as pd
import requests
import json
from keybert import KeyBERT
from wikidata.client import Client

from data import get_dataset


def text_to_graph_wikidata(
        N, #numero di salti
        text, #domanda
        ):

    kw_model = KeyBERT()
    kw = kw_model.extract_keywords(text)
    txt = [kw[i][0].lower() for i in range(len(kw))]
    txt = np.unique(txt)
    ids = [get_id(word) for word in txt]
    txt_id = ['<id_word> ' + ids[i] + ' <word> ' + txt[i] + ' <desc> ' + extract_info_node(ids[i])[1] for i in range(len(txt)) if ids[i] is not None]
    triplets_list = []
    entities_list = txt_id

    for i in range(N):
        # Get all entities for each text

        triplets = extract_triplets(txt_id)
        #triplets = [(triplet[0], triplet[1], triplet[2]) for triplet in triplets]
        triplets = np.unique(triplets, axis=0)
        triplets_list.extend([triplet for triplet in triplets])

        entities = [triplet[2] for triplet in triplets]
        entities = [entity for entity in entities if entity not in entities_list]
        txt_id = np.unique(entities)

        entities_list = np.hstack((entities_list, txt_id))

    return triplets_list

def get_id(text):

    url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&search=" + text
    response = requests.get(url)
    data = json.loads(response.text)
    if data['search']:
        id = data['search'][0]['id']
        return id
    else:
        return None

def extract_triplets(txt_ids):

    triplets = []

    for ktxt in txt_ids:

        wid = ktxt.split('<id_word>')[1].split('<word>')[0].strip() #word id
        url = "https://www.wikidata.org/w/api.php?action=wbgetclaims&format=json&entity=" + wid
        response = requests.get(url)
        data = json.loads(response.text)
        for item in data['claims']:
            if item.startswith("P"):
                predicate = item
                claims = data['claims'][item][0]
                if "mainsnak" in claims:
                    if "datavalue" in claims["mainsnak"]:
                        if "value" in claims["mainsnak"]["datavalue"]:
                            value = claims['mainsnak']['datavalue']['value']
                            if type(value) is dict and 'id' in value:

                                objects = value['id']
                                info_pred = extract_info_node(predicate)
                                predicate = '<id_word> ' + predicate + ' <word> ' + info_pred[0] + ' <desc> ' + info_pred[1]
                                info_obj = extract_info_node(objects)
                                objects = '<id_word> ' + objects + ' <word> ' + info_obj[0] + ' <desc> ' + info_obj[1]
                                triplets.append((ktxt, predicate, objects))

    return triplets


def extract_info_node(node_id):

    client = Client()
    item = client.get(node_id)
    description = str(item.description)
    name = str(item.label)

    return [name, description]

### da modificare ###
def print_info_triples(triples):

    for s, r, p in triples:
        try:
            extract_info_node(s)
            extract_info_node(r)
            extract_info_node(p)
            print('-------------------------------')
        except:
            print('Error with ' + str(s))


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

    rel = 'rel'
    subj = 'arg1'
    obj = 'arg2'

    graph = get_dataset('conceptnet')
    graph = graph['train']
    graph = graph.to_pandas()

    triplets_list = []
    entities_list = kw

    for i in range(N):

        filtered = graph.loc[graph[subj].isin(kw)][[subj, rel, obj]]
        triplets = filtered.to_numpy()
        triplets = [(item[0], item[1], item[2]) for item in triplets]
        triplets = np.unique(triplets, axis=0)
        triplets_list.extend([triplet for triplet in triplets])

        kw = filtered.loc[~filtered[obj].isin(entities_list)][obj].drop_duplicates().to_numpy()

        entities_list = np.hstack((entities_list, kw))

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


def create_memory(model, graph, args):

    sentences = list(graph.keys())
    embeddings = {}
    # Loop through each sentence in the list
    for sentence in sentences:
        # Encode the sentence into a 384 dimensional vector
        embedding = model.encode(sentence, **args)
        # Store the embedding in the dictionary with the sentence as the key
        embeddings[sentence] = embedding
    return embeddings


"""
def rel_to_adj(relations):
    #controllare nuovo esito se giusto
    #pdb.set_trace()
    g = {k: vs for k, vs in relations.items() if vs is not None}
    edges = [(a, b) for k, bs in g.items() for a, b in bs]
    df = pd.DataFrame(edges)
    if df.shape[0] > 500000:
        chunk_size = 500000
        chunks = [x for x in range(0, df.shape[0], chunk_size)]
        adj = pd.concat([pd.crosstab(df.iloc[chunks[i]:chunks[i + 1], 0], df.iloc[chunks[i]:chunks[i + 1], 1]) for i in range(0, len(chunks) - 1)])
        adj_dict = {}
        if adj.loc[adj.index.duplicated()].shape[0] != 0:
            key = adj.index
            for k in key:
                if k in adj.index[adj.index.duplicated()]:
                    adj_dict[k] = adj.loc[k].sum()
                else:
                    adj_dict[k] = adj.loc[k]
            A = pd.DataFrame.from_dict(adj_dict, orient='index')
        else:
            A = adj

    else:
        A = pd.crosstab(df[0], df[1])

    A[A.isna()] = 0
    A[A >= 2] = 1

    return A
"""




