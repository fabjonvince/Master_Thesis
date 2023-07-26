import pdb

import nltk
import torch
from datasets import load_metric
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np

from data import get_dataset
from preprocess import from_triplets_of_ids_to_triplets_of_string


class SingleReasoningPath:
    def __init__(self, root_node, topk):
        self.kpath = list()
        self.root_node = root_node
        for _ in range(topk):
            self.kpath.append([('ROOT_NODE', root_node, 1.)])
        self.topk = topk

    def add_new_step(self, rel, node, prob, k):
        self.kpath[k].append((rel, node, prob))

    def get_current_nodes(self):
        nodes = list()
        for kp in self.kpath:
            nodes.append((kp[-1][1], kp[-1][2]))
        return nodes

    def get_root_node(self):
        return self.root_node

    def get_reasoning_path(self):
        return self.kpath


class AllReasoningPath:
    def __init__(self):
        self.all_path = dict()

    def set_root_nodes(self, root_nodes, topk):
        self.all_path = dict()
        for node in root_nodes:
            path = SingleReasoningPath(node, topk)
            self.all_path[node] = path

    def get_current_nodes(self, root_node=None):
        if root_node:
            return self.all_path[root_node].get_current_nodes()
        else:
            return {k: v.get_current_nodes() for k, v in self.all_path.items()}

    def add_new_step(self, root_node, k, rel, node, prob):
        self.all_path[root_node].add_new_step(rel, node, prob, k)

    def get_all_reasoning_path(self):
        return {k: v.get_reasoning_path() for k, v in self.all_path.items()}


# Define a function named find_triplets. It will be used inside the Custom layer
def find_triplets(list_of_triplets, start=None, rel=None, end=None):
    # Initialize an empty list to store the matching triplets
    result = []
    # Loop through each triplet in the input list
    for triplet in list_of_triplets:
        # Check if the triplet matches the start, rel, end parameters
        # If any parameter is None, it means any value is acceptable
        if (start is None or triplet[0] == start) and (rel is None or triplet[1] == rel) and (
                end is None or triplet[2] == end):
            # Add the matching triplet to the result list
            result.append(triplet)
    # Return the result list
    return result


def extract_all_relations_for_a_node(node_name, triplets):
    return np.unique([r for _, r, _ in find_triplets(triplets, start=node_name)]).tolist()


def extract_values_from_tensor(tensor, indices):
    """
    Extracts values from a tensor based on the given indices.

    Arguments:
    tensor -- A torch.Tensor of shape (M, N) containing real numbers.
    indices -- A list of lists specifying the indices for each row of the tensor.

    Returns:
    result -- A list of lists of tensors containing values from the tensor
              corresponding to the given indices.
    """

    result = []

    # Iterate over the rows of the tensor and corresponding indices
    for row_indices in indices:
        row_result = []

        # Extract values from the row based on the given indices
        for index in row_indices:
            row_result.append(tensor[index])

        result.append(torch.stack(row_result))

    return result


def get_rouge_scores(references, predictions):
    rouge_metric = load_metric("rouge")

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in references]
    rouge_scores = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_scores_final = {key: value.mid.fmeasure * 100 for key, value in rouge_scores.items()}
    rouge_scores_final = {k: round(v, 2) for k, v in rouge_scores_final.items()}
    rouge_scores_final['R1_prec'] = rouge_scores['rouge1'].mid.precision * 100
    rouge_scores_final['R'] = (rouge_scores['rouge1'].mid.fmeasure * 100 + rouge_scores['rouge2'].mid.fmeasure * 100 +
                               rouge_scores['rougeL'].mid.fmeasure * 100) / 3
    return rouge_scores_final


def get_rouge_scores_for_hftrainer(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return get_rouge_scores(decoded_labels, decoded_preds)


def get_bert_scores(predictions, references):
    bertscore = SingletonBertmetric()
    result_bs = bertscore.compute(predictions=predictions, references=references, idf=False, batch_size=32,
                                  lang="en",
                                  rescale_with_baseline=True, model_type="microsoft/deberta-large-mnli")
    results = {'BS_' + k: (sum(v) / len(v)) * 100 for k, v in result_bs.items() if k in ['precision', 'recall', 'f1']}
    return results


def get_bartscore(predictions, sources):
    with torch.no_grad():
        bart_scorer = SingletonBartScorer()
        score = bart_scorer.score(sources, predictions, batch_size=4)
    return sum(score) / len(score)


class SingletonBartScorer(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = BARTScorer()
        return cls.instance


class SingletonBertmetric(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = load_metric("bertscore")
        return cls.instance

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))


def find_kg_pathes(start, end, kg:list, max_distance=3):
    if max_distance == 0:
        return None
    triplets = find_triplets(kg, end=end)
    if len(triplets) == 0:
        return None

    final_trip = find_triplets(triplets, start=start)
    if len(final_trip) != 0:
        return [final_trip[0]]

    for triplet in triplets:
        new_end_node = triplet[0]
        final_trip = find_kg_pathes(start, new_end_node, kg, max_distance-1)
        if final_trip is not None:
            final_trip.append(triplet)
            return final_trip


def create_oracle_graph(row, ids_to_nodes, ids_to_rels):
    pdb.set_trace()
    keysq = row['keywords']
    keysa = row['answer_keyword']
    graph = row['graph']
    if graph[-3:] != 'npy':  # add the extension if it is not present
        graph = graph + '.npy'
    graph = np.load(graph)  # the graph contains triplets of int that are indices of nodes and rels

    kg = from_triplets_of_ids_to_triplets_of_string(graph, ids_to_nodes, ids_to_rels)  # convert the triplets of ids to triplets of string

    graphs = dict()
    for keyq in keysq:
        kgraphs = list()
        for keya in keysa:
            kgraph = find_kg_pathes(keyq, keya, kg)
            if kgraph is not None:
                kgraphs.append(kgraph)
        graphs[keyq] = kgraphs
    return graphs


