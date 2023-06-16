import pdb

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import tensor

from preprocess import load_with_pickle, from_triplets_of_ids_to_triplets_of_string
from tools import AllReasoningPath, get_rouge_scores, get_bert_scores


gen_val_params = {
    'max_length': 128,
    'num_beams': 2,
    'no_repeat_ngram_size': 3,
    'early_stopping': True,
    'length_penalty': 1.1,
    'repetition_penalty': 1.5,
    'min_length': 100,
}

gen_test_params = {
    'max_length': 128,
    'num_beams': 4,
    'no_repeat_ngram_size': 3,
    'early_stopping': True,
    'length_penalty': 1.1,
    'repetition_penalty': 1.5,
    'min_length': 100,
}

class GNNQA(pl.LightningModule):
    def __init__(self, model=None,
                 memory_rels:pd.DataFrame=None,
                 memory_nodes:pd.DataFrame=None,
                 node_embs=None,
                 tokenizer=None,
                 save_dir=None,
                 model_lr=None,
                 gnn_lr=None,
                 gnn_layers=None):
        super().__init__()
        if gnn_layers is None:
            gnn_layers = []
        self.gnn_layers = gnn_layers
        self.model = model
        self.memory_rels = memory_rels
        self.memory_nodes = memory_nodes
        self.memory_rels.set_index('custom_index', inplace=True)
        self.memory_nodes.set_index('custom_index', inplace=True)
        self.nodes_dict = self.memory_nodes['custom_value'].to_dict()
        self.rels_dict = self.memory_rels['custom_value'].to_dict()
        self.memory_embs = node_embs
        self.model_lr = model_lr
        self.gnn_lr = gnn_lr
        self.tokenizer = tokenizer
        self.val_metric = []
        self.test_metrics = {}
        self.save_dir = save_dir

    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                gnn_triplets=None,
                gnn_mask=None,
                rel_mask=None,
                current_reasoning_path=None,
                memory_nodes=None,
                rels_ids=None,
                ):

        print('Forward step')

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, gnn_triplets=gnn_triplets,
                            gnn_mask=gnn_mask, rel_mask=rel_mask, current_reasoning_path=current_reasoning_path,
                            memory_nodes=memory_nodes, rels_ids=rels_ids)

        return output.loss, output.logits

    def prepare_data_from_batch(self, batch):
        toks = \
            self.tokenizer(batch['T5_question'], padding='max_length', truncation=True, max_length=128,
                           return_tensors='pt').to(self.device)
        input_ids, attention_mask = toks['input_ids'], toks['attention_mask']
        answer = batch['answers']['text']
        if len(answer) > 1:
            answer = [answer[0]]
        labels = self.tokenizer(answer, padding='max_length', truncation=True, max_length=512, return_tensors='pt')[
            'input_ids'].to(self.device)
        # labels = tensor(labels, dtype=torch.long)
        graph = batch['graph'] # the graph contain the path to the file containing the graph
        graph = np.load(graph) # the graph contains triplets of int that are indices of nodes and rels
        pdb.set_trace()
        graph = from_triplets_of_ids_to_triplets_of_string(graph, self.nodes_dict, self.rels_dict)
        rels_ids = self.memory_rels.Index
        batch['rel_mask'] = (input_ids == 32100).int()
        batch['gnn_mask'] = (input_ids == 32101).int()
        keywords = batch['keywords']
        reasoning_path = AllReasoningPath()
        reasoning_path.set_root_nodes(keywords, 2)
        return batch, input_ids, attention_mask, labels, graph, reasoning_path, rels_ids

    def training_step(self, batch, batch_idx):
        #pdb.set_trace()
        batch, input_ids, attention_mask, labels, graph, reasoning_path, rels_ids = self.prepare_data_from_batch(batch)

        loss = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, gnn_triplets=graph,
                    gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'], current_reasoning_path=reasoning_path,
                    memory_nodes=self.memory_nodes, rels_ids=rels_ids)[0]

        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, input_ids, attention_mask, labels, graph, reasoning_path, rels_ids = self.prepare_data_from_batch(batch)

        predictions = self.model.generate(input_ids=input_ids, gnn_triplets=graph,
                                   gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'],
                                   current_reasoning_path=reasoning_path,
                                   memory_nodes=self.memory_nodes,
                                   rels_ids=rels_ids, **gen_val_params)
        predictions = [self.tokenizer.decode(predictions[0], skip_special_tokens=True)]
        targets = batch['answers']['text']
        if len(targets) > 1:
            targets = [targets[0]]
        val_metric = get_rouge_scores(predictions, targets)
        self.val_metric.append(val_metric['R'])
        return

    def test_step(self, batch, batch_idx):
        batch, input_ids, attention_mask, labels, graph, reasoning_path, rels_ids = self.prepare_data_from_batch(batch)

        predictions = self.model.generate(input_ids=input_ids, gnn_triplets=graph,
                                          gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'],
                                          current_reasoning_path=reasoning_path,
                                          memory_nodes=self.memory_nodes,
                                          rels_ids=rels_ids, **gen_test_params)
        predictions = [self.tokenizer.decode(predictions[0], skip_special_tokens=True)]
        targets = batch['answers']['text']
        if len(targets) > 1:
            targets = [targets[0]]
        test_metric = get_rouge_scores(predictions, targets)
        test_bs = get_bert_scores(predictions, targets)
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

        if not 'question' in self.test_metrics:
            self.test_metrics['question'] = []
        self.test_metrics['question'].append(batch['T5_question'])
        if not 'target_answer' in self.test_metrics:
            self.test_metrics['target_answer'] = []
        self.test_metrics['target_answer'].append(targets[0])
        if not 'predicted_answer' in self.test_metrics:
            self.test_metrics['predicted_answer'] = []
        self.test_metrics['predicted_answer'].append(predictions[0])
        if not 'graph' in self.test_metrics:
            self.test_metrics['graph'] = []
        self.test_metrics['graph'].append(self.model.encoder.get_and_clean_reasoning_path().get_all_reasoning_path())
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
        self.log('val_rouge', sum(self.val_metric)/len(self.val_metric))
        self.val_metric = []



    def configure_optimizers(self):
        #pdb.set_trace()
        if self.gnn_lr:
            gnn_parameters = []
            model_parameters = []
            layers = ['encoder.block.{}.'.format(i) for i in self.gnn_layers]
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



