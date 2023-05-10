import pdb

import pytorch_lightning as pl
import torch
from torch import tensor



class GNNQA(pl.LightningModule):
    def __init__(self, model=None, rel_model=None, nodes_model=None):
        super().__init__()
        self.model = model
        self.rel_model = rel_model #sentence transformer
        self.nodes_model = nodes_model  # sentence transformer


    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                graph=None,
                edges=None,
                rel=None,
                enc_rel=None,
                adj=None,
                ):

        print('Forward step')
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, graph=graph, edges=edges, rel=rel, enc_rel=enc_rel, adj=adj)

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):

        print('training step ')
        #'q_id = id domanda
        #title = question
        #selftext = text with additional information
        #document = vuoto
        #subreddit = direttive output es. explain like im five
        #answers
        #title_urls = url, vuoto
        #selftext_urls = url, vuoto
        #answers_urls = url delle risposte
        #answer_tok = tokenizzate answer

        input_ids = batch['input_ids']
        input_ids = tensor(input_ids, dtype=torch.int, device=self.device)
        attention_mask = batch['attention_mask']
        attention_mask = tensor(attention_mask, dtype=torch.int, device=self.device)
        labels = batch['answer_tok']['input_ids']

        #passare al layer KIL ->

        # dizionario con chiave parola nella domanda e valore=indice/i della parola nel testo

        graph = batch['graph']

        relations = batch['relations']
        rel = {k: vs for k, vs in relations.items() if vs is not None}
        enc_rel = []
        # applicare sentence transformer a rel
        for key in rel.keys():
            enc_rel.append(self.rel_model.encode(rel[key]))


        edges = batch['edges']
        edges = {k: vs for k, vs in edges.items() if vs is not None}
        enc_edges = []
        # applicare sentence transformer a nodes
        for key in edges.keys():
            enc_edges.append(self.nodes_model.encode(edges[key]))

        adj = batch['adj']


        loss = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, graph=graph, edges=enc_edges, rel=relations, enc_rel=enc_rel, adj=adj)[0]

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class T5DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, dataset, batch_size=1, args=None, name_mapping=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.args = args

        dataset_columns = name_mapping.get(args.dataset, None)
        self.train_name = dataset_columns[0]
        self.eval_name = dataset_columns[1]
        self.test_name = dataset_columns[2]
        self.question_name = dataset_columns[3]
        self.answers_name = dataset_columns[4]



    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset[self.train_name], batch_size=self.batch_size)

    """
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['validation'], batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['test'], batch_size=self.batch_size)"""


