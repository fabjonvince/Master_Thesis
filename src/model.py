import pytorch_lightning as pl
import torch

class GNNQA(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        #self.hparams = hparams

    def forward(self, x):
        print('forward step')
        return x

    def training_step(self, batch, batch_idx):
        print('training step')
        return {"loss": torch.tensor(1.0)}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def test_step(self, batch, batch_idx):
        return {"loss": torch.tensor(1.0)}


class CustomKilLayer(torch.nn.Module):
    def __init__(self,
                 n_rel, # Numero di tutte le possibili relazioni
                 edges,
                 ):
        print('KIL layer')
        super().__init__()
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


