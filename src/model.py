import pdb

import pytorch_lightning as pl
import torch
from torch import tensor

from tools import AllReasoningPath, get_rouge_scores


class GNNQA(pl.LightningModule):
    def __init__(self, model=None, memory_rels=None, memory_nodes=None,  tokenizer=None): # model_lr=None, gnn_lr=None,
        super().__init__()
        self.model = model
        self.memory_rels = memory_rels
        self.memory_nodes = memory_nodes
        #self.model_lr = model_lr
        #self.gnn_lr = gnn_lr
        self.tokenizer = tokenizer
        self.val_metric = []

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
        toks = \
            self.tokenizer(batch['question'], padding='max_length', truncation=True, max_length=128,
                           return_tensors='pt').to(self.device)
        input_ids, attention_mask = toks['input_ids'], toks['attention_mask']
        labels = self.tokenizer(batch['answers']['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')['input_ids'].to(self.device)
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

        '''
        predictions = self.model.generate(input_ids=input_ids, gnn_triplets=graph,
                                   gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'],
                                   current_reasoning_path=reasoning_path,
                                   memory_nodes=self.memory_nodes,
                                   rels_ids=rels_ids)
        predictions = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
        targets = batch['answers']['text']
        val_metric = get_rouge_scores(predictions, batch['title'])#targets)
        self.val_metric.extend(val_metric['R'])
        return
        
        '''

        loss = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, gnn_triplets=graph,
                    gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'], current_reasoning_path=reasoning_path,
                    memory_nodes=self.memory_nodes, rels_ids=rels_ids)[0]

        return loss

    '''
    def on_validation_epoch_end(self):
        self.log('val_rouge', sum(self.val_metric)/len(self.val_metric))
        self.val_metric = []

    '''

    def configure_optimizers(self):
        #opt1 = torch.optim.SGD(self.parameters(), lr=self.model_lr)
        #opt2 = torch.optim.SGD(self.parameters(), lr=self.model_lr)
        return torch.optim.SGD(self.parameters(), lr=0.00001)



