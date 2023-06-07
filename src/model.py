import pdb

import pytorch_lightning as pl
import torch
from sentence_transformers import SentenceTransformer
from torch import tensor

from tools import AllReasoningPath


class GNNQA(pl.LightningModule):
    def __init__(self, model=None, memory_rels=None, memory_nodes=None):
        super().__init__()
        self.model = model
        self.memory_rels = memory_rels
        self.memory_nodes = memory_nodes

    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                graph=None,
                gnn_mask=None,
                rel_mask=None,
                current_reasoning_path=None,
                nodes=None,
                rels=None,
                keywords=None,
                rels_ids=None,
                model_lr=None,
                gnn_lr=None,
                ):

        print('Forward step')

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, graph=graph,
                            gnn_mask=gnn_mask, rel_mask=rel_mask, current_reasoning_path=current_reasoning_path,
                            nodes=nodes, rels=rels, keywords=keywords, rels_ids=rels_ids)

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):

        print('training step ')

        input_ids = batch['input_ids']
        input_ids = tensor(input_ids, dtype=torch.int, device=self.device)
        attention_mask = batch['attention_mask']
        attention_mask = tensor(attention_mask, dtype=torch.int, device=self.device)
        labels = batch['answer_tok']['input_ids']
        graph = batch['graph']
        rels_ids = {k: v for v, k in enumerate(self.memory_rels.keys())}
        batch['rel_mask'] = (batch['input_ids'] == 32100).int()
        batch['gnn_mask'] = (batch['input_ids'] == 32101).int()
        keywords = batch['keywords']
        reasoning_path = AllReasoningPath()
        reasoning_path.set_root_nodes(keywords, 2)

        loss = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, graph=graph,
                    gnn_mask=batch['gnn_mask'], rel_mask=batch['rel_mask'], current_reasoning_path=reasoning_path,
                    nodes=self.memory_nodes, rels=self.memory_rels, keywords=keywords, rels_ids=rels_ids)[0]

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)



