import pdb
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import pickle
from preprocess import from_triplets_of_ids_to_triplets_of_string
from tools import AllReasoningPath, get_rouge_scores, get_bert_scores, get_bartscore, find_kg_pathes
import nltk

gen_val_params = {
    'max_length': 140,
    'num_beams': 2,
    'no_repeat_ngram_size': 3,
    'early_stopping': True,
    'length_penalty': 1.1,
    'repetition_penalty': 1.5,
    'min_length': 50,
}

gen_test_params = {
    'max_length': 140,
    'num_beams': 3,
    'no_repeat_ngram_size': 3,
    'early_stopping': True,
    'length_penalty': 1.1,
    'repetition_penalty': 1.5,
    'min_length': 50,
}

class GNNQA(pl.LightningModule):
    def __init__(self, model=None,
                 ids_to_rels=None,
                 ids_to_nodes=None,
                 memory_embs=None,
                 tokenizer=None,
                 save_dir=None,
                 model_lr=None,
                 gnn_lr=None,
                 gnn_layers=None,
                 labels=None,
                 use_support_document=False,
                 create_embeddings_with_model=False,
                 use_oracle_graphs=False,
                 ):
        super().__init__()
        if gnn_layers is None:
            gnn_layers = []
        self.gnn_layers = gnn_layers
        self.model = model

        # dictionary containing k:v where k is the node/rel id and v is the node/rel value
        self.ids_to_rels = ids_to_rels
        self.rels_to_ids = {v: k for k, v in ids_to_rels.items()}
        self.ids_to_nodes = ids_to_nodes
        self.nodes_to_ids = {v: k for k, v in ids_to_nodes.items()}
        # dictionary containing node embeddings
        self.memory_embs = memory_embs

        self.model_lr = model_lr
        self.gnn_lr = gnn_lr
        self.tokenizer = tokenizer
        self.val_metric = {}
        self.test_metrics = {}
        self.save_dir = save_dir
        self.labels = labels
        self.use_support_document = use_support_document
        if self.use_support_document == True:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": tokenizer.additional_special_tokens + ["<SUPP_DOC_TOK>"]})
        self.use_oracle_graphs = use_oracle_graphs
        if self.use_oracle_graphs == True:
            # I set stop word using NLTK
            self.stop_words = nltk.corpus.stopwords.words('english')
        self.create_embeddings_with_model = create_embeddings_with_model

    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                gnn_triplets=None,
                gnn_mask=None,
                rel_mask=None,
                current_reasoning_path=None,
                memory_embs=None,
                rels_ids=None,
                model_lr=None,
                gnn_lr=None,
                ):

        #print('Forward step')
        #with torch.autograd.set_detect_anomaly(True):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, gnn_triplets=gnn_triplets,
                            gnn_mask=gnn_mask, rel_mask=rel_mask, current_reasoning_path=current_reasoning_path,
                            memory_embs=memory_embs, rels_ids=rels_ids)

        return output.loss, output.logits

    def get_all_pathes(self, keywords, answer, kg, max_path_length=3):
        end_nodes = answer.lower().split(' ')
        end_nodes = [e for e in end_nodes if e not in self.stop_words]
        all_pathes = []
        for keyword in keywords:
            for end_node in end_nodes:
                path = find_kg_pathes(keyword, end_node, kg, max_path_length)
                if path is not None:
                    all_pathes.append(path)
        return all_pathes

    # retrieve data from the batch for the next step
    def prepare_data_from_batch(self, batch, skip_oracle_graph_creation=False):
        #pdb.set_trace()

        if self.use_support_document == True and batch['support_documents'] != '':
            support_documents = batch['support_documents']
            toks = \
                self.tokenizer(batch['question'] + ' <SUPP_DOC_TOK> ' + support_documents, padding=True, truncation=True, max_length=1024,
                           return_tensors='pt').to(self.device)

        else:
            toks = \
                self.tokenizer(batch['question'], padding=True, truncation=True, max_length=128,
                               return_tensors='pt').to(self.device)

        input_ids, attention_mask = toks['input_ids'], toks['attention_mask']
        if len(self.labels.split(',')) > 1:
            answer = batch[self.labels.split(',')[0]][self.labels.split(',')[1]]
        else:
            answer = batch[self.labels]
        # I need to check if answer is a list with more than 1 value
        if type(answer) == list:
            if len(answer) > 1:
                answer = [answer[0]]
        else:
            answer = [answer]
        labels = self.tokenizer(answer, padding=True, truncation=True, return_tensors='pt')[
            'input_ids'].to(self.device)
        # labels = tensor(labels, dtype=torch.long)
        graph = batch['graph'] # the graph contain the path to the file containing the graph
        if graph[-3:] != 'npy': # add the extension if it is not present
            graph = graph + '.npy'
        graph = np.load(graph) # the graph contains triplets of int that are indices of nodes and rels

        graph = from_triplets_of_ids_to_triplets_of_string(graph, self.ids_to_nodes, self.ids_to_rels) # convert the triplets of ids to triplets of string
        rels_ids = {v: k for k,v in self.ids_to_rels.items()} # reverse the dictionary
        batch['rel_mask'] = (input_ids == 32100).int()
        batch['gnn_mask'] = (input_ids == 32101).int()
        keywords = batch['keywords']
        # set the path in the graph
        reasoning_path = AllReasoningPath()
        reasoning_path.set_root_nodes(keywords, 2)

        all_pathes = None
        if not skip_oracle_graph_creation:
            if self.use_oracle_graphs:
                if 'oracle_graphs' in batch:
                    path = batch['oracle_graphs']
                    with open(path, 'rb') as f:
                        all_pathes = pickle.load(f).to_dict()
                else:
                    #pdb.set_trace()
                    if type(answer) == list:
                        answer = answer[0]
                    all_pathes = self.get_all_pathes(keywords, answer, graph, max_path_length=3)

        return batch, input_ids, attention_mask, labels, graph, reasoning_path, rels_ids, all_pathes

    def prepare_for_oracle_graph_training(self, all_paths, topk=2):
        """
        This function turn all paths into a dictionary where K are the root_words and the value are the probability distributions for each step.
        Two dictionry one for relations and one for nodes
        Args:
            all_pathes:

        Returns:

        """
        if all_paths is None:
            raise Exception('all_pathes is None')
        else:
            rel_target_dict = {}
            node_target_dict = {}
            for k, paths in all_paths.items():
                rel_targets = []
                node_targets = []
                paths = paths[:topk] # I select the topK path
                for i in range(len(paths[0])):
                    rs = [path[i][1] for path in paths]
                    rids = [self.rels_to_ids[r] for r in rs]
                    len_r = len(self.ids_to_rels)
                    rel_target = torch.zeros(len_r, dtype=self.dtype)
                    value = 1 / len(rids)
                    for rid in rids:
                        rel_target[rid] = rel_target[rid] + value
                    rel_target = rel_target.to(self.device)
                    rel_targets.append(rel_target)

                    # now the last nodes
                    ns = [path[i][2] for path in paths]
                    nids = [self.nodes_to_ids[n] for n in ns]
                    node_target = [(r,k,v) for k,r,v in zip(ns, rs, nids)]
                    # len_n = len(self.nodes_to_ids)
                    # node_target = torch.zeros(len_n)
                    # value = 1 / len(nids)
                    # for nid in nids:
                    #     node_target[nid] = node_target[nid] + value
                    # node_target = node_target.to(self.device)
                    node_targets.append(node_target)
                rel_target_dict[k] = rel_targets
                node_target_dict[k] = node_targets
            return rel_target_dict, node_target_dict

    def prepare_target_graphs(self, all_paths, k):
        """
        This function generates from the oracle paths the path to force into the model during train.
        The K indicates the topK path that model generates from each root_word. Can happen that the number of path is less than K,
        in this case I repeat the path until I have K paths
        Args:
            all_paths:
            k:

        Returns:

        """
        target_paths = {}
        for key, paths in all_paths.items():
            if len(paths) > k:
                """
                This should be useless because I already select the topK paths
                """
                # I select k path at random
                paths = random.sample(paths, k)
            else:
                # I select all the paths, and repeat them until I have k paths
                paths = paths * (k // len(paths)) + paths[:k % len(paths)]
            target_paths[key] = paths
        return target_paths

    def training_step(self, batch, batch_idx):
        batch, input_ids, attention_mask, labels, graph, reasoning_path, rels_ids, all_pathes = self.prepare_data_from_batch(batch)
        if self.create_embeddings_with_model:
            self.generate_embeddings(graph)
        if self.use_oracle_graphs:
            pdb.set_trace()
            # The graph should contain topk paths for each root_word
            # Now I fix larger
            all_pathes = {k: v[:self.model.args.gnn_topk] for k, v in all_pathes.items()}
            # now I fix smaller
            for k, v in all_pathes.items():
                if len(v) < self.model.args.gnn_topk:
                    v = v * (self.model.args.gnn_topk // len(v)) + v[:self.model.args.gnn_topk % len(v)]
                    all_pathes[k] = v

            # target is a dictionary where k are keywords and v are probability distribution, one for each triplet in the all_paths.
            targets_rel, targets_node = self.prepare_for_oracle_graph_training(all_pathes)
            targets = self.prepare_target_graphs(all_pathes, self.model.args.gnn_topk)
            reasoning_path.set_targets(targets_rel, targets_node, targets)


            #pdb.set_trace()


        loss = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, gnn_triplets=graph,
                    gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'], current_reasoning_path=reasoning_path,
                    memory_embs=self.memory_embs, rels_ids=rels_ids)[0]

        if self.use_oracle_graphs:
            rel_loss, node_loss = reasoning_path.loss_rels, reasoning_path.loss_nodes
            if rel_loss is not None:
                # rel_loss is a list of scalar tensor and I want to convert it into a tensor
                rel_loss = [r for r in rel_loss if r is not None]
                rel_loss = torch.stack(rel_loss)
                rel_loss = torch.mean(rel_loss)
                self.log('train_loss_rel', rel_loss.item(), on_step=True, on_epoch=False, prog_bar=True)
                loss = loss + rel_loss
            if node_loss is not None:
                node_loss = [n for n in node_loss if n is not None]
                node_loss = torch.stack(node_loss)
                node_loss = torch.mean(node_loss)
                self.log('train_loss_node', node_loss.item(), on_step=True, on_epoch=False, prog_bar=True)
                loss = loss + node_loss

        self.log('train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #pdb.set_trace()
        batch, input_ids, attention_mask, labels, graph, reasoning_path, rels_ids, _ = self.prepare_data_from_batch(batch, True)
        if self.create_embeddings_with_model:
            self.generate_embeddings(graph)

        with torch.no_grad():
            predictions = self.model.generate(input_ids=input_ids, gnn_triplets=graph,
                                       gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'],
                                       current_reasoning_path=reasoning_path,
                                       memory_embs=self.memory_embs,
                                       rels_ids=rels_ids, **gen_val_params)
        predictions = [self.tokenizer.decode(predictions[0], skip_special_tokens=True)]

        if len(self.labels.split(',')) > 1:
            targets = batch[self.labels.split(',')[0]][self.labels.split(',')[1]]
        else:
            targets = batch[self.labels]
        if type(targets) == list:
            if len(targets) > 1:
                targets = [targets[0]]
        else:
            targets = [targets]

        val_metric = get_rouge_scores(predictions, targets)
        val_bs = get_bert_scores(predictions, targets)
        val_bart = get_bartscore(predictions, targets)

        for k,v in val_metric.items():
            if k in self.val_metric:
                self.val_metric[k].append(v)
            else:
                self.val_metric[k] = [v]

        for k,v in val_bs.items():
            if k in self.val_metric:
                self.val_metric[k].append(v)
            else:
                self.val_metric[k] = [v]
        if 'bart_score' not in self.val_metric:
            self.val_metric['bart_score'] = []
        self.val_metric['bart_score'].append(val_bart)

        if not 'question' in self.val_metric:
            self.val_metric['question'] = []
        self.val_metric['question'].append(batch['question'])
        if not 'target_answer' in self.val_metric:
            self.val_metric['target_answer'] = []
        self.val_metric['target_answer'].append(targets[0])
        if not 'predicted_answer' in self.val_metric:
            self.val_metric['predicted_answer'] = []
        self.val_metric['predicted_answer'].append(predictions[0])
        if not 'graph' in self.val_metric:
            self.val_metric['graph'] = []
        self.val_metric['graph'].append(self.model.get_and_clean_reasoning_path().get_all_reasoning_path())
        
        return

    def test_step(self, batch, batch_idx):
        #pdb.set_trace()
        batch, input_ids, attention_mask, labels, graph, reasoning_path, rels_ids, _ = self.prepare_data_from_batch(batch, True)
        if self.create_embeddings_with_model:
            self.generate_embeddings(graph)

        with torch.no_grad():
            predictions = self.model.generate(input_ids=input_ids, gnn_triplets=graph,
                                              gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'],
                                              current_reasoning_path=reasoning_path,
                                              memory_embs=self.memory_embs,
                                              rels_ids=rels_ids, **gen_test_params)
        predictions = [self.tokenizer.decode(predictions[0], skip_special_tokens=True)]
        if len(self.labels.split(',')) > 1:
            targets = batch[self.labels.split(',')[0]][self.labels.split(',')[1]]
        else:
            targets = batch[self.labels]
        if type(targets) == list:
            if len(targets) > 1:
                targets = [targets[0]]
        else:
            targets = [targets]

        test_metric = get_rouge_scores(predictions, targets)
        test_bs = get_bert_scores(predictions, targets)
        test_bart = get_bartscore(predictions, targets)
        for k,v in test_metric.items():
            if k in self.test_metrics:
                self.test_metrics[k].append(v)
            else:
                self.test_metrics[k] = [v]

        for k,v in test_bs.items():
            if k in self.test_metrics:
                self.test_metrics[k].append(v)
            else:
                self.test_metrics[k] = [v]
        if 'bart_score' not in self.test_metrics:
            self.test_metrics['bart_score'] = []
        self.test_metrics['bart_score'].append(test_bart)

        if not 'question' in self.test_metrics:
            self.test_metrics['question'] = []
        self.test_metrics['question'].append(batch['question'])
        if not 'target_answer' in self.test_metrics:
            self.test_metrics['target_answer'] = []
        self.test_metrics['target_answer'].append(targets[0])
        if not 'predicted_answer' in self.test_metrics:
            self.test_metrics['predicted_answer'] = []
        self.test_metrics['predicted_answer'].append(predictions[0])
        if not 'graph' in self.test_metrics:
            self.test_metrics['graph'] = []
        self.test_metrics['graph'].append(self.model.get_and_clean_reasoning_path().get_all_reasoning_path())
        return

    def on_test_epoch_end(self):
        for k,v in self.test_metrics.items():
            if not k in ['question', 'target_answer', 'predicted_answer', 'graph']:
                self.log(k, sum(v)/len(v), prog_bar=True)
        if not self.save_dir is None:
            table = pd.DataFrame(self.test_metrics)
            table.to_csv(self.save_dir + '/test_results.csv', index=False)
        self.test_metrics = {}

    def on_validation_epoch_end(self):
        for k,v in self.val_metric.items():
            if not k in ['question', 'target_answer', 'predicted_answer', 'graph']:
                if k=='R':
                    self.log('val_rouge', sum(v)/len(v), prog_bar=True)
                else:
                    self.log(k, sum(v)/len(v), prog_bar=True)
        if not self.save_dir is None:
            table = pd.DataFrame(self.val_metric)
            table.to_csv(self.save_dir + '/val_e' + str(self.current_epoch) + '.csv', index=False)
        self.val_metric = {}



    def configure_optimizers(self):
        #pdb.set_trace()
        if self.gnn_lr:
            gnn_parameters = []
            model_parameters = []
            layers = ['encoder.block.{}.layer.2'.format(i) for i in self.gnn_layers]
            for k,v in self.model.named_parameters():
                if any(x in k for x in layers):
                    print('GNN Layer added to second optimizer', k, v.shape)
                    gnn_parameters.append(v)
                else:
                    model_parameters.append(v)
            opt = torch.optim.AdamW([{
                'params': model_parameters,
                'lr': self.model_lr
            }, {
                'params': gnn_parameters,
                'lr': self.gnn_lr,
            }])
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=self.model_lr)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                "scheduler": torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.7),
                "interval": "epoch",
            }
        }

    # prende in input, lista di nodi, modello per embedding
    def generate_embeddings(self, graph, batch_size=64):

        nodes = []
        nodes.extend([s for s, _, _ in graph] + [e for _, _, e in graph])
        nodes = list(set(nodes))

        self.memory_embs = {}
        # tokenize all the nodes together to take advantage of parallel processing of the GPU
        node_tok = self.tokenizer(nodes, padding='max_length', truncation=True, max_length=32,
                                  return_tensors='pt')['input_ids'].to(self.device)

        # then loop through the batches of tokens and compute the embeddings
        with torch.no_grad():
            for i in range(0, len(nodes), batch_size):
                selected_nodes_tokens = node_tok[i: i + batch_size]
                embedded = self.model.shared(selected_nodes_tokens)
                for j, node in enumerate(nodes[i: i + batch_size]):
                    self.memory_embs[node] = torch.mean(embedded[j], dim=0).detach().cpu()
        return


