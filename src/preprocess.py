import requests
import json

"""
def text_to_graph2(N, text1, text2):
    # Get all entities for each text
    entities1 = get_entities(text1)
    entities2 = get_entities(text2)

    # Get all relations for each entity
    relations1 = get_relations(entities1, N)
    relations2 = get_relations(entities2, N)

    # Merge common nodes
    merged_nodes = merge_nodes(relations1, relations2)

    # Convert to triplets
    triplets = convert_to_triplets(merged_nodes)

    return triplets
"""

def text_to_graph(N, text):
    # Get all entities for each text
    entities = get_entities(text)

    # Get all relations for each entity
    relations = get_relations(entities, N)

    # Convert to triplets
    triplets = convert_to_triplets(relations)

    return triplets


def get_entities(text):
    url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&search=" + text
    response = requests.get(url)
    data = json.loads(response.text)
    entities = []
    for item in data['search']:
        entities.append(item['id'])
    return entities


def get_relations(entities, N):
    relations = []
    for entity in entities:
        url = "https://www.wikidata.org/w/api.php?action=wbgetclaims&format=json&entity=" + entity
        response = requests.get(url)
        data = json.loads(response.text)
        cont = 0
        for item in data['claims']:
            if item.startswith("P"):
                relation = item[1:]
                if relation not in relations:
                    relations.append(relation)
                    if N > 1:

                        claims = data['claims'][item][0]

                        if "mainsnak" in claims:
                            if "datavalue" in claims["mainsnak"]:
                                if "value" in claims["mainsnak"]["datavalue"]:
                                    value = claims['mainsnak']['datavalue']['value']

                                    if type(value) is dict and "id" in value :
                                        sub_entities = get_entities(value['id'])
                                        sub_relations = get_relations(sub_entities, N - 1)
                                        for sub_relation in sub_relations:
                                            if sub_relation not in relations:
                                                relations.append(sub_relation)

    return relations


def merge_nodes(relations1, relations2):
    merged_nodes = []
    for relation in relations1:
        if relation in relations2:
            merged_nodes.append(relation)
            relations2.remove(relation)
        else:
            merged_nodes.append(relation + "_1")
    for relation in relations2:
        merged_nodes.append(relation + "_2")
    return merged_nodes


def convert_to_triplets(nodes):
    triplets = []
    for i in range(len(nodes)):
        if i % 3 == 0:
            entity1 = nodes[i]
            entity2 = nodes[i + 2]
            relation = nodes[i + 1]
            triplets.append((entity1, relation, entity2))
    return triplets


def print_triplets(triplets):
    for triplet in triplets:
        print(triplet[0] + " -> " + triplet[1] + " -> " + triplet[2])
