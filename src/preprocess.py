import pdb
import re

import numpy as np
import pandas as pd
import requests
import json
from keybert import KeyBERT
from wikidata.client import Client


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



def add_special_tokens(
        question, #domanda
        ):

    kw_model = KeyBERT()
    kw = kw_model.extract_keywords(question)
    txt = [kw[i][0].lower() for i in range(len(kw))]
    txt = np.unique(txt)

    new_question = question
    for word in txt:
        idx = re.search(r"\b({})\b".format(word), new_question, re.IGNORECASE).start()
        new_question = new_question[:idx] + "<REL_TOKEN> " + new_question[idx:]

    return new_question


def text_to_graph_concept(
        N, #numero di salti
        text, #domanda
        graph, #grafo
        subj, #first arg
        ):

    text = text.split()
    txt = [re.sub("[^a-z]", "", text[i + 1].lower()) for i in range(len(text)) if text[i] == "<REL_TOKEN>"]

    triplets_list = []
    entities_list = txt

    for i in range(N):

        triplets = graph.loc[graph[subj].isin(txt)][['arg1', 'rel', 'arg2']].to_numpy()
        triplets = [(item[0], item[1], item[2]) for item in triplets]
        triplets = np.unique(triplets, axis=0)
        triplets_list.extend([triplet for triplet in triplets])

        entities = [triplet[2] for triplet in triplets]
        entities = [entity for entity in entities if entity not in entities_list]
        txt = np.unique(entities)

        entities_list = np.hstack((entities_list, txt))

    return triplets_list


def print_triplets(triplets):
    for triplet in triplets:
        print(triplet[0] + " -> " + triplet[1] + " -> " + triplet[2])


def graph_to_rel(triplets):
    relations = {}
    nodes = np.unique([item for rel in triplets for item in [rel[0], rel[2]]])

    for rel in triplets:

        if rel[1] in relations.keys():
            relations[rel[1]].append((rel[0], rel[2]))
        else:
            relations[rel[1]] = [(rel[0], rel[2])]

    for node in nodes:

        if "self_rel" in relations.keys():
            relations["self_rel"].append((node, node))
        else:
            relations["self_rel"] = [(node, node)]

    return relations


def graph_to_nodes(triplets):

    edges = {}

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

    return edges


def rel_to_adj(relations):
    #controllare nuovo esito se giusto
    g = {k: vs for k, vs in relations.items() if vs is not None}
    edges = [(a, b) for k, bs in g.items() for a, b in bs]
    df = pd.DataFrame(edges)
    A = pd.crosstab(df[0], df[1])
    A[A>=2] = 1

    return A
