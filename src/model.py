import pdb

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import tensor

from tools import AllReasoningPath, get_rouge_scores, get_bert_scores


class GNNQA(pl.LightningModule):
    def __init__(self, model=None, memory_rels=None, memory_nodes=None,  tokenizer=None, save_dir=None): # model_lr=None, gnn_lr=None,
        super().__init__()
        self.model = model
        self.memory_rels = memory_rels
        self.memory_nodes = memory_nodes
        #self.model_lr = model_lr
        #self.gnn_lr = gnn_lr
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

    def training_step(self, batch, batch_idx):
        #pdb.set_trace()
        toks = \
            self.tokenizer(batch['question'], padding='max_length', truncation=True, max_length=128,
                           return_tensors='pt').to(self.device)
        input_ids, attention_mask = toks['input_ids'], toks['attention_mask']
        answer = batch['answers']['text']
        if len(answer) > 1:
            answer = [answer[0]]
        labels = self.tokenizer(answer, padding='max_length', truncation=True, max_length=512, return_tensors='pt')['input_ids'].to(self.device)
        #labels = tensor(labels, dtype=torch.long)
        graph = batch['graph']
        rels_ids = {k: v for v, k in enumerate(self.memory_rels.keys())}
        batch['rel_mask'] = (input_ids == 32100).int()
        batch['gnn_mask'] = (input_ids == 32101).int()
        keywords = batch['keywords']
        reasoning_path = AllReasoningPath()
        reasoning_path.set_root_nodes(keywords, 2)

        loss = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, gnn_triplets=graph,
                    gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'], current_reasoning_path=reasoning_path,
                    memory_nodes=self.memory_nodes, rels_ids=rels_ids)[0]

        return loss

    def validation_step(self, batch, batch_idx):
        toks = \
        self.tokenizer(batch['question'], padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(self.device)
        input_ids, attention_mask = toks['input_ids'], toks['attention_mask']
        # attention_mask = tensor(attention_mask, dtype=torch.int)
        labels = self.tokenizer(batch['answers']['text'], padding='max_length', truncation=True, max_length=512,
                                return_tensors='pt')['input_ids'].to(self.device)
        # labels = tensor(labels, dtype=torch.long)
        graph = batch['graph']

        rels_ids = {k: v for v, k in enumerate(self.memory_rels.keys())}
        batch['rel_mask'] = (input_ids == 32100).int()
        batch['gnn_mask'] = (input_ids == 32101).int()
        keywords = batch['keywords']
        reasoning_path = AllReasoningPath()
        reasoning_path.set_root_nodes(keywords, 2)

        predictions = self.model.generate(input_ids=input_ids, gnn_triplets=graph,
                                   gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'],
                                   current_reasoning_path=reasoning_path,
                                   memory_nodes=self.memory_nodes,
                                   rels_ids=rels_ids)
        predictions = [self.tokenizer.decode(predictions[0], skip_special_tokens=True)]
        targets = batch['answers']['text']
        if len(targets) > 1:
            targets = [targets[0]]
        val_metric = get_rouge_scores(predictions, targets)
        self.val_metric.append(val_metric['R'])
        return

    def test_step(self, batch, batch_idx):
        toks = \
            self.tokenizer(batch['question'], padding='max_length', truncation=True, max_length=128,
                           return_tensors='pt').to(self.device)
        input_ids, attention_mask = toks['input_ids'], toks['attention_mask']
        # attention_mask = tensor(attention_mask, dtype=torch.int)
        labels = self.tokenizer(batch['answers']['text'], padding='max_length', truncation=True, max_length=512,
                                return_tensors='pt')['input_ids'].to(self.device)
        # labels = tensor(labels, dtype=torch.long)
        graph = batch['graph']

        rels_ids = {k: v for v, k in enumerate(self.memory_rels.keys())}
        batch['rel_mask'] = (input_ids == 32100).int()
        batch['gnn_mask'] = (input_ids == 32101).int()
        keywords = batch['keywords']
        reasoning_path = AllReasoningPath()
        reasoning_path.set_root_nodes(keywords, 2)

        predictions = self.model.generate(input_ids=input_ids, gnn_triplets=graph,
                                          gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'],
                                          current_reasoning_path=reasoning_path,
                                          memory_nodes=self.memory_nodes,
                                          rels_ids=rels_ids)
        predictions = [self.tokenizer.decode(predictions[0], skip_special_tokens=True)]
        targets = batch['answers']['text']
        if len(targets) > 1:
            targets = [targets[0]]
        test_metric = get_rouge_scores(predictions, targets)
        test_bs = get_bert_scores(predictions, targets)
        for k,v in test_metric.values():
            if k in self.test_metrics:
                self.test_metrics[k].append(v)
            else:
                self.test_metrics[k] = [v]

        for k,v in test_bs.values():
            if k in self.test_metrics:
                self.test_metrics[k].append(v)
            else:
                self.test_metrics[k] = [v]

        if not 'question' in self.test_metrics:
            self.test_metrics['question'] = []
        self.test_metrics['question'].append(batch['question'][0])
        if not 'target_answer' in self.test_metrics:
            self.test_metrics['target_answer'] = []
        self.test_metrics['target_answer'].append(targets[0])
        if not 'predicted_answer' in self.test_metrics:
            self.test_metrics['predicted_answer'] = []
        self.test_metrics['predicted_answer'].append(predictions[0])
        if not 'graph' in self.test_metrics:
            self.test_metrics['graph'] = []
        self.test_metrics['graph'].append(self.model.encoder.get_and_clean_reasoning_path())
        return

    def test_epoch_end(self, outputs):
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
        #opt1 = torch.optim.SGD(self.parameters(), lr=self.model_lr)
        #opt2 = torch.optim.SGD(self.parameters(), lr=self.model_lr)
        return torch.optim.SGD(self.parameters(), lr=0.00001)



