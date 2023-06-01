import pdb
import re

import numpy as np
import pandas as pd
import requests
import json
from keybert import KeyBERT


def text_to_graph(
        N, #numero di salti
        text, #domanda,
        debug=False
        ):

    if debug:
        pdb.set_trace()

    kw_model = KeyBERT()
    kw = kw_model.extract_keywords(text)
    txt = [kw[i][0].lower() for i in range(len(kw))]
    txt = np.unique(txt)

    triplets_list = []
    entities_list = txt

    for i in range(N):
        # Get all entities for each text
        entities = get_entities(txt, N=N)

        triplets = convert_to_triplets(entities)
        triplets = [(triplet[0].lower(), triplet[1].lower(), triplet[2].lower()) for triplet in triplets]
        triplets = np.unique(triplets, axis=0)
        triplets_list.extend([triplet for triplet in triplets])

        entities = [triplet[2] for triplet in triplets]
        entities = [entity for entity in entities if entity not in entities_list]
        txt = np.unique(entities)
        entities_list = np.hstack((entities_list, txt))

    return triplets_list

def get_entities(text, N, cont=0, debug=False):

    entities = []
    if debug:
        pdb.set_trace()
    if cont == 0:
        for word in text:
            url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&search=" + word
            response = requests.get(url)
            data = json.loads(response.text)
            if 'search' in data:
                for item in data['search']:
                    if item['label'] not in entities:
                        new_item = re.sub(r'[\W+]', '', str(item['label']))
                        if new_item != "":
                            sub_entities = get_entities(item['label'], N-1, cont + 1)
                            entities.append([word, [item['label'], sub_entities]])

    else:

        url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&search=" + text
        response = requests.get(url)
        data = json.loads(response.text)

        if 'search' in data:
            for item in data['search']:
                if item['label'] not in entities:
                    new_item = re.sub(r'[\W+]', '', str(item['label']))
                    if new_item != "":
                        entities.append(item['label'])

        entities = np.unique(entities)

    return entities


def convert_to_triplets(nodes):
    # index 0 = source, altri index = relazioni
    triplets = []
    for i in range(len(nodes)):
        entity1 = nodes[i][0]
        relation = nodes[i][1][0]

        for j in range(len(nodes[i][1][1])):
            triplets.append((entity1, relation, nodes[i][1][1][j]))

    return triplets


def print_triplets(triplets):
    for triplet in triplets:
        print(triplet[0] + " -> " + triplet[1] + " -> " + triplet[2])


def graph_to_nodes(triplets):
    nodes = np.unique([item for rel in triplets for item in rel])

    return nodes


def graph_to_rel(triplets):
    relations = {}
    nodes = np.unique([item for rel in triplets for item in rel])

    for rel in triplets:

        if rel[1] in relations.keys():
            relations[rel[1]].append((rel[0], rel[2]))
        else:
            relations[rel[1]] = [(rel[0], rel[2])]

    for node in nodes:
        #vedere se dà errore e fare assegnazione prima volta
        if "self_rel" in relations.keys():
            relations["self_rel"].append((node, node))
        else:
            relations["self_rel"] = [(node, node)]

    return relations


def graph_to_edges(triplets):

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
