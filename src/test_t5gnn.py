import argparse

import numpy as np
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer

from tools import AllReasoningPath
from t5 import T5GNNForConditionalGeneration

argparser = argparse.ArgumentParser()





argparser.add_argument('--layer_with_gnn', nargs='+', type=int, default=[1, 2], help='Layers with KIL')
argparser.add_argument('--gnn_topk', type=int, default=2, help='Number of topk nodes to consider for each root node')
name_mapping = {
"eli5": ("train_eli5", "validation_eli5", "test_eli5", "title", "answers"),
"conceptnet": ("rel", "arg1", "arg2"),
}
args = argparser.parse_args()
setattr(args, 'n_rel', 24)
setattr(args, 'gnn_embs_size', 384)

gnn_model = T5GNNForConditionalGeneration.from_pretrained('t5-base', args)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.add_special_tokens({"additional_special_tokens": ["<REL-TOK>", "<GNN-TOK>"]})
questions = ['why <REL-TOK><GNN-TOK>water is a <REL-TOK><GNN-TOK>fluid?']
gnn_triplets = [('water', 'is', 'fluid'), ('water', 'hasproperty', 'trasparent'), ('water', 'isusedfor', 'drink'), ('water', 'is', 'h2o'), ('water', 'madeof', 'hydrogen'), ('fluid', 'is', 'material state'), ('hydrogen', 'is', 'chemical element')]
selected_words=['water', 'fluid']
answer = ['I don\'t know']

labels_ids = tokenizer(answer, return_tensors='pt')['input_ids']
batch = tokenizer(questions, return_tensors='pt', max_length=16, padding='max_length')
batch['rel_mask'] = (batch['input_ids'] == 32100).int()
batch['gnn_mask'] = (batch['input_ids'] == 32101).int()
batch['gnn_triplets'] = gnn_triplets
batch['decoder_input_ids'] = labels_ids


# Load a pretrained model with all-MiniLM-L12-v2 checkpoint
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Define a lists of sentences to encode, nodes and relations
nodes = np.unique([start for start, _, _ in gnn_triplets] + [end for _,_, end in gnn_triplets])
gnn_triplets.extend([(node, 'self', node) for node in nodes])
rels = np.unique([rel for  _, rel, _ in gnn_triplets])

print('Extracted nodes')
print(nodes)

print('Extracted rels')
print(rels)



def create_memory(model, sentences, args):
  embeddings={}
  # Loop through each sentence in the list
  for sentence in sentences:
    # Encode the sentence into a 384 dimensional vector
    embedding = model.encode(sentence, **args)
    # Store the embedding in the dictionary with the sentence as the key
    embeddings[sentence] = embedding
  return embeddings

memory_nodes = create_memory(model, nodes, {'convert_to_tensor': True})
memory_rels = create_memory(model, rels, {'convert_to_tensor': True})
batch['memory_nodes'] = memory_nodes

rels_ids = {k: v for v, k in enumerate(memory_rels.keys()) }
print(rels_ids)
nodes_ids = {k: v for v, k in enumerate(memory_nodes.keys())}
batch['rels_ids'] = rels_ids


rp = AllReasoningPath()

# now I create two reasoning paths starting from water and drink. From each of them I save the two most probable reasoning paths
rp.set_root_nodes(['water', 'fluid'], 2)
batch['current_reasoning_path'] = rp
gnn_model(**batch)

