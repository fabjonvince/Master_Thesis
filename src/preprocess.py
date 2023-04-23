import pdb

import numpy as np
import requests
import json
from keybert import KeyBERT


def text_to_graph(
        N, #numero di salti
        text #domanda
        ):

    #pdb.set_trace()
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

def get_entities(text, N, cont=0):

    entities = []
    if cont == 0:
        for word in text:
            url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&search=" + word
            response = requests.get(url)
            data = json.loads(response.text)
            if 'search' in data:
                for item in data['search']:
                    if item['label'] not in entities:
                        sub_entities = get_entities(item['label'], N-1, cont + 1)
                        entities.append([word, [item['label'], sub_entities]])

    else:
        url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&search=" + text
        response = requests.get(url)
        data = json.loads(response.text)

        if 'search' in data:
            for item in data['search']:
                if item['label'] not in entities:
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

    for rel in triplets:
        if rel[0] in relations.keys():
            if rel[1] not in relations[rel[0]]:
                relations.update({rel[0]: np.append(relations[rel[0]], rel[1])})
        else:
            relations[rel[0]] = np.array(rel[1])

        if rel[1] in relations.keys():
            if rel[1] not in relations[rel[0]]:
                relations.update({rel[1]: np.append(relations[rel[1]], rel[0])})
                relations.update({rel[1]: np.append(relations[rel[1]], rel[2])})
        else:
            relations[rel[1]] = np.array(rel[0])
            relations.update({rel[1]: np.append(relations[rel[1]], rel[2])})

        if rel[2] in relations.keys():
            if rel[1] not in relations[rel[0]]:
                relations.update({rel[2]: np.append(relations[rel[2]], rel[1])})
        else:
            relations[rel[2]] = np.array(rel[1])

    return relations




