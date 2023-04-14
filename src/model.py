import pytorch_lightning as pl
import torch
from torch import Tensor, tensor


class GNNQA(pl.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                graph=None
                ):

        print('Forward step')
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, graph=graph)
        print('bbbbbbbbbbbbbbbb')
        exit()
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
        input_ids = tensor(input_ids, dtype=torch.int, device='cuda:0')
        #input_ids = Tensor(input_ids)
        attention_mask = batch['attention_mask']
        attention_mask = tensor(attention_mask, dtype=torch.int, device='cuda:0')
        labels = batch['answer_tok']['input_ids']
        graph = batch['graph']

        loss = self(input_ids, attention_mask, labels, graph)[0]

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

        self.dataset = self.dataset.map(lambda example: self.tokenizer(example[self.answers_name]['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'))
        self.dataset = self.dataset.map(lambda example: {'answer_tok': self.tokenizer(example[self.question_name], padding='max_length', truncation=True, max_length=512, return_tensors='pt')})


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

    """
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['validation'], batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset['test'], batch_size=self.batch_size)"""


"""

class CustomKilLayer(torch.nn.Module):
    def __init__(self,
                 n_rel, # Numero di tutte le possibili relazioni
                 edges,
                 ):
        super().__init__()
        print('KIL layer')
        self.register_parameter("wrel", torch.nn.Parameter(torch.Tensor(1, n_rel)))

    def customCRW(
            self,
            A: torch.Tensor,  # shape [Nrel X NNodes X NNodes]
            node_index: int,  # indice del nodo corrente
            prel: torch.Tensor,  # shape[NNodes X Nrel X 1]
            tprev: torch.Tensor,  # shape[NNodes X NNodes X 1]
            wrel: torch.Tensor,  # shape[NRel X 1]

    ):
        Ac = torch.sum((wrel[None, :] * prel[node_index]).squeeze() * A, dim=0)  # shape NNodes X NNodes
        D = torch.diag(torch.sum(Ac, dim=-1, keepdim=True).squeeze())  # shape NNodes X NNodes
        M = torch.linalg.inv(D) * Ac  # NNodes X NNodes
        t = tprev * M
        return t

    def relation_pred(
            self,
            inputs,  # words embedding   shape [Nwords X dim_embeddings = 768]
            rels,  # relation of every embedding   shape [Nrel X dim_embeddings]
    ):
        ### relation prediction ###

        # project reletion token's embedding into key memory
        lin_proj = torch.nn.Linear(inputs.size(-1), inputs.size(-1))
        q = lin_proj(inputs)  # torch.nn.Linear()

        # normalization
        qn = torch.nn.LayerNorm(q)

        # dot-product similarity
        dp = torch.dot(qn, rels)

        # apply softmax
        sftm = torch.nnSoftmax(dim=1)
        distrib = sftm(dp)  # shape [Nwords X Nrelation X 1]

        return distrib

    def knowledge_integration(
            self,
            inputs,
            node_embds,
            residual_embds
    ):
        ### knowledge-injected LM ###

        # compute residual connection
        res_conn = torch.dot(inputs, node_embds)

        # use value projection block
        lin_proj = torch.nn.Linear(inputs.size(-1), inputs.size(-1))
        V = lin_proj(res_conn, residual_embds)

        # add the trasformed embedding
        Vlm = residual_embds + V

        # normalization
        Vn = torch.nn.LayerNorm(Vlm)

        return Vn

    def forward(self,
                inputs_embeds, # embeddings of all the tokens in the sentece
                token_index, # index of the token
                node_index, # index of the root node
                edges, # embeddings of all the nodes
                A,
                rels = None, # relations embeddings
                ):
        # token_embs shape [Nwords X dim_embeddings = 768]
        # edges shape [Nwords X Nwords X Nrelation]
        # A shape [Nrelation X Nwords X Nwords]

        # relation prediction
        prels = self.relation_pred(inputs_embeds[token_index], rels)
        t = self.customCRW(A, node_index, prels, self.tprev, self.wrel)
        self.tprev = t
        new_embds = self.knowledge_integration(t, edges, edges[node_index])
        return new_embds

"""
