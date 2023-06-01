import pdb

import pytorch_lightning as pl
import torch
from sentence_transformers import SentenceTransformer
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
                nodes=None,
                enc_nodes=None,
                rel=None,
                enc_rel=None,
                adj=None,
                ):

        print('Forward step')
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, graph=graph, nodes=nodes, enc_nodes=enc_nodes, rel=rel, enc_rel=enc_rel, adj=adj)

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):

        print('training step ')

        input_ids = batch['input_ids']
        input_ids = tensor(input_ids, dtype=torch.int, device=self.device)
        attention_mask = batch['attention_mask']
        attention_mask = tensor(attention_mask, dtype=torch.int, device=self.device)
        labels = batch['answer_tok']['input_ids']
        graph = batch['graph']
        nodes = batch['nodes']
        relations = batch['relations']
        adj = batch['adj']

        nodes_dict = {k: vs for k, vs in nodes.items() if vs is not None}

        # enc_nod = {}
        # checkpoint = input("Insert checkpoint to use: ")  # checkpoint for sentence transformer, separated by space
        # checkpoint = checkpoint.split(' ')
        # for mc in checkpoint:

        model_sent = SentenceTransformer('all-MiniLM-L6-v2')
        model_sent.max_seq_length = 12
        nodes_enc = [model_sent.encode(nodes, batch_size=128, convert_to_tensor=True) for nodes in nodes_dict.values()]
        enc_nod = [{key: nodes_enc[i]} for i, key in enumerate(nodes_dict.keys())]

        """
        'paraphrase-MiniLM-L3-v2 47.77237868309021', 
        'paraphrase-albert-small-v2 55.20635676383972', 
        'all-MiniLM-L6-v2 69.34488487243652', 

        'paraphrase-MiniLM-L3-v2 49.446166038513184',
        'paraphrase-albert-small-v2 54.45325207710266',
        'all-MiniLM-L6-v2 67.73390364646912',

        'paraphrase-MiniLM-L3-v2 49.50322461128235', 
        'paraphrase-albert-small-v2 54.93287134170532',
        'all-MiniLM-L6-v2 66.79923272132874'

        #  paraphrase-MiniLM-L3-v2 paraphrase-albert-small-v2 all-MiniLM-L6-v2
        #     speed      smallest size   fast and good quality
        """

        rel_dict = {k: vs for k, vs in relations.items() if vs is not None}
        model_sent = SentenceTransformer('all-MiniLM-L6-v2')
        model_sent.max_seq_length = 12
        rel_enc = [model_sent.encode(rel, batch_size=128, convert_to_tensor=True) for rel in rel_dict.values()]
        enc_rel = [{key: rel_enc[i]} for i, key in enumerate(rel_dict.keys())]

        adj_dict = {k: vs for k, vs in adj.items() if vs is not None}


        loss = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, graph=graph, nodes=nodes_dict, enc_nodes=enc_nod, rel=rel_dict, enc_rel=enc_rel, adj=adj_dict)[0]

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


