import copy
import pdb
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers import T5PreTrainedModel, T5Config
from torch import nn, Tensor, tensor
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, \
    T5LayerCrossAttention, T5Stack
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from tools import extract_all_relations_for_a_node, extract_values_from_tensor, find_triplets, AllReasoningPath


#from model import CustomKilLayer


class CustomKilLayer(torch.nn.Module):

    def __init__(self,
                 n_rel, # Numero di tutte le possibili relazioni
                 n_nodes):
        super().__init__()
        self.register_parameter("wrel", torch.nn.Parameter(torch.Tensor(1, n_rel)))
        self.tprev = torch.zeros(n_nodes, n_nodes)

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
        lin_proj = torch.nn.Linear(inputs.size(-1), inputs.size(-1), device='cuda:0')
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
                    hidden_states,
                    attention_mask=None,
                    position_bias=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    encoder_decoder_position_bias=None,
                    layer_head_mask=None,
                    cross_attn_layer_head_mask=None,
                    past_key_value=None,
                    use_cache=None,
                    output_attentions=None,
                    token_index=None,  # index of the token
                    node_index=None,  # index of the root node
                    graph=None, # graph of the sentence
                    edges=None, # embeddings of all the nodes
                    rel=None, # relations
                    enc_rel=None, # relations embeddings
                    adj=None, # adjacency matrix
                ):


        self.execute_oreolm(
            inputs_embeds=hidden_states,
            token_index=token_index,
            node_index=node_index,
            graph=graph,
            edges=edges,
            adj=adj,
            rel=rel
            )


    def execute_oreolm (self,
                        inputs_embeds, # embeddings of all the tokens in the sentence
                        token_index, # index of the token
                        node_index, # index of the root node
                        graph, # graph of the sentence
                        edges, # embeddings of all the nodes, Vent
                        adj,
                        rel # relations embeddings
                    ):
        # token_embs shape [Nwords X dim_embeddings = 768]
        # edges shape [Nwords X Nwords X Nrelation]
        # A shape [Nrelation X Nwords X Nwords]

        print('oreolm layer')

        # relation prediction
        prels = self.relation_pred(inputs_embeds[token_index], rel)
        # contextualized random walk
        t = self.customCRW(adj, node_index, prels, self.tprev, self.wrel)
        self.tprev = t
        # knowledge integration
        new_embds = self.knowledge_integration(t, edges, edges[node_index])

        return new_embds


class CustomGNNLayer(torch.nn.Module):

    def __init__(self,
                 n_rel,  # Numero di tutte le possibili relazioni
                 embs_size,
                 model_size,
                 topk=2,
                 ):
        super().__init__()
        self.embs_size = embs_size
        self.model_size = model_size
        self.n_rel = n_rel
        self.topk = topk
        # the classification head is composed by a linear layer and a softmax function
        self.classification_head = torch.nn.Sequential(torch.nn.Linear(self.model_size, self.n_rel),
                                                       torch.nn.Softmax(dim=1))

        # Now I need layer to perform a attention reprojection
        self.query_reprj = torch.nn.Sequential(torch.nn.Linear(self.model_size, self.model_size), torch.nn.Tanh())
        self.nodes_reprj = torch.nn.Sequential(torch.nn.Linear(self.embs_size, self.model_size), torch.nn.Tanh())

        # Now the reprojection layer to inject graph knowledge into the GNN-TOK
        self.gnn_reprj = torch.nn.Sequential(torch.nn.Linear(self.embs_size, self.model_size), torch.nn.Tanh())

    def calculate_scores(self, query, k_nodes, probabilities):
        """
        Calculates the final scores for each embedding in each group based on the given query, groups, and probabilities.

        Args:
            query (torch.Tensor): The embedding of the query text of shape (1, Q).
            k_nodes (list of torch.Tensor): A list of N groups of node embeddings, where each group is of shape (M, Q).
            probabilities (torch.Tensor): A tensor of size (1, N) containing the probabilities assigned to each group.

        Returns:
            scores (list of lists): A list of lists containing the final scores for each embedding in each group.
        """

        # Pad the groups to make them equally sized
        max_size = max(len(group) for group in k_nodes)
        groups_padded = [F.pad(group, (0, 0, 0, max_size - len(group))) for group in k_nodes]

        # Stack the padded groups and place on the same device of the model
        groups_stacked_tmp = torch.stack(groups_padded).to(next(self.parameters()).device)

        mask = (groups_stacked_tmp != 0).int()[:, :, 0]

        # reprojection of nodes
        groups_stacked = self.nodes_reprj(groups_stacked_tmp)

        # reprojection of query
        query = self.query_reprj(query)

        # Expand dimensions to match batch sizes for broadcasting
        query_expanded = query.unsqueeze(0)
        probabilities_expanded = probabilities.unsqueeze(2)

        # Calculate the dot product between the query and each embedding in the groups
        dot_products = torch.matmul(groups_stacked, query_expanded.permute(0, 2, 1))

        # Apply softmax normalization to the dot products
        attention_weights = F.softmax(dot_products, dim=1)
        # print(attention_weights.shape)
        # print(probabilities_expanded.shape)

        # Weight the attention weights by the probability assigned to the group
        logits = attention_weights * probabilities_expanded.view(attention_weights.size(0), 1, 1) / 0.1
        # print(logits.shape)
        weighted_attention = F.softmax(logits.view(-1, ), dim=0)
        # print(weighted_attention)
        weighted_attention = weighted_attention.view(*logits.shape)

        weighted_attention[mask == 0] = 0
        #print("wa", weighted_attention.shape)

        return weighted_attention, groups_stacked_tmp

    def forward(self,
                hidden_states,  # embeddings of all the tokens in the sentence
                gnn_mask=None,  # index of the gnn tokens
                rel_mask=None,  # index of the rel tokens
                gnn_triplets=None,  # list of triplets
                memory_nodes=None,  # node memory
                current_reasoning_path: AllReasoningPath = None,
                rels_ids=None,  # ids of the relations in the memory
                ):

        # I generate the probability over all the relations
        rel_prob = self.classification_head(hidden_states[rel_mask.bool()])
        # rel_prob shape (batch_size=1, n_REL_TOK, n_rels)

        current_nodes = current_reasoning_path.get_current_nodes()
        # now current nodes is a dict where k are the root nodes and v are a list of topk elements representing the current node

        queries = hidden_states[gnn_mask.bool()]
        output = list()
        for query, probs, root_word in zip(queries, rel_prob, current_nodes.keys()):
            # I select the last node in the reasoning path
            nodes = [n for n, _ in current_nodes[root_word]]

            # I extract their relations creating a list
            noderels = [(k, extract_all_relations_for_a_node(k, gnn_triplets)) for k in nodes]

            # I turn relations into ids
            noderelsids = [[rels_ids[r] for r in v] for k, v in noderels]

            # I turn relation into probability
            pnoderels = extract_values_from_tensor(probs, noderelsids)

            # Now I create the groups. A group is a set of triplets that has the same start, and rel.
            # group_nodes contains only the end nodes in a structure equal to the pnoderels.
            # in this way for each prob in pnoderels corresponds a node in groups_nodes
            groups_nodes = list()
            selected_nodes = list()

            for (k, rels), probs, ids in zip(noderels, pnoderels, noderelsids):
                groups_nodes.append([[n for _, _, n in find_triplets(gnn_triplets, start=k, rel=rel)] for rel in rels])

            all_scores = list()

            # now it computes the weighted contribution of all the ends nodes.
            # They are weighted by the relation weights and the score computed with an attention mechanism
            for nods, probs, rels, cur_node, k in zip(groups_nodes, pnoderels, noderels, nodes, range(self.topk)):
                # nods contains all the end nodes that have cur_node as start node
                node_embs = list()
                for ns in nods:
                    # turn them into embedding using the memory matrix
                    node_embs.append(torch.stack([torch.tensor(memory_nodes[n]) for n in ns]))
                # compute the scores associaed to each embedding
                scores, embs = self.calculate_scores(query.view(1, -1), k_nodes=node_embs,
                                                     probabilities=probs.view(1, -1))

                scores = scores.view(scores.size(0) * scores.size(1), -1)
                embs = embs.view(scores.size(0) * scores.size(1), -1)
                #print('scores embs', scores.shape, embs.shape)
                # weight embs for the scores
                weighted_embs = embs * scores
                #print('weighted_embs', weighted_embs.shape)
                # Compute the mean of the weighted embeddings
                mean_emb = torch.mean(weighted_embs, dim=0)
                #print('mean_emb', mean_emb.shape)

                # Now I have to add the topk end nodes to the reasoning path
                # first I create a list containing tuple (rel, nodes) which are all the possibile end nodes from the current node
                node_rel_list = [(r, n) for r, ns in zip(rels[1], nods) for n in ns]

                # now update the reasoning path. The same current node can lead to the same edge nodes.
                # to avoid that in case the end node is already selected we use the next most probable end node.
                keep = True
                tmp_scores = scores[scores != 0].view(1, -1)
                value, indices = torch.sort(tmp_scores, descending=True)
                value = value.view(-1, ).tolist()
                indices = indices.view(-1, ).tolist()
                index = 0
                next_node = None

                #print(indices)
                #print(value)

                while keep:
                    # then I extract the index and the end node with the highest probability
                    next_node = node_rel_list[indices[index]]
                    if next_node in selected_nodes and index < len(node_rel_list) - 1:
                        # next_node already selected
                        index += 1
                    else:
                        selected_nodes.append(next_node)
                        keep = False

                # Add the most probable node to the reasoning path
                current_reasoning_path.add_new_step(root_word, k, next_node[0], next_node[1], value[index])
                all_scores.append(mean_emb)
            all_scores = torch.stack(all_scores).mean(dim=0)
            output.append(all_scores)
        output = torch.stack(output)
        output = self.gnn_reprj(output)
        # Now I update the hidden states

        hidden_states[gnn_mask.bool()] = hidden_states[gnn_mask.bool()] + output

        return hidden_states, current_reasoning_path

class T5GNNBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))
        self.layer.append(CustomGNNLayer(
            n_rel=config.n_rel, embs_size=config.gnn_embs_size, model_size=config.d_model, topk=config.gnn_topk
        ))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        gnn_mask=None,  # index of the gnn tokens
        rel_mask=None,  # index of the rel tokens
        gnn_triplets=None,  # list of triplets
        memory_nodes=None,  # node memory
        current_reasoning_path: AllReasoningPath = None,
        rels_ids=None,  # ids of the relations in the memory
    ):

        if past_key_value is not None:
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-2](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        # Apply GNN layer
        hidden_states, current_reasoning_path = self.layer[-1](
            hidden_states,  # embeddings of all the tokens in the sentence
            gnn_mask,  # index of the gnn tokens
            rel_mask,  # index of the rel tokens
            gnn_triplets,  # list of triplets
            memory_nodes,  # node memory
            current_reasoning_path,
            rels_ids,  # ids of the relations in the memory
            )

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5GNNStack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        #self.block = nn.ModuleList(
        #    [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        #)
        layers = list()
        for i in range(config.num_layers):
            if i in config.layer_with_gnn:
                layers.append(T5GNNBlock(config, has_relative_attention_bias=bool(i == 0)))
            else:
                layers.append(T5Block(config, has_relative_attention_bias=bool(i == 0)))
        self.block = nn.ModuleList(layers)
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False



    def parallelize(self, device_map=None):

        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gnn_mask=None,  # index of the gnn tokens
        rel_mask=None,  # index of the rel tokens
        gnn_triplets=None,  # list of triplets
        memory_nodes=None,  # node memory
        current_reasoning_path: AllReasoningPath = None,
        rels_ids=None,  # ids of the relations in the memory
    ):

        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):


            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs, **kwargs):
                        return tuple(module(*inputs, use_cache, output_attentions, **kwargs))

                    return custom_forward

                if isinstance(layer_module, T5Block):
                    layer_outputs = checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        extended_attention_mask,
                        position_bias,
                        encoder_hidden_states,
                        encoder_extended_attention_mask,
                        encoder_decoder_position_bias,
                        layer_head_mask,
                        cross_attn_layer_head_mask,
                        None,  # past_key_value is always None with gradient checkpointing
                    )
                else:
                    layer_outputs = checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        extended_attention_mask,
                        position_bias,
                        encoder_hidden_states,
                        encoder_extended_attention_mask,
                        encoder_decoder_position_bias,
                        layer_head_mask,
                        cross_attn_layer_head_mask,
                        None,  # past_key_value is always None with gradient checkpointing
                        gnn_mask=gnn_mask,  # index of the gnn tokens
                        rel_mask=rel_mask,  # index of the rel tokens
                        gnn_triplets=gnn_triplets,  # list of triplets
                        memory_nodes=memory_nodes,  # node memory
                        current_reasoning_path=current_reasoning_path,  # current reasoning path
                        rels_ids=rels_ids,  # ids of the relations in the memory
                    )

            else:
                if isinstance(layer_module, T5GNNBlock):
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=extended_attention_mask,
                        position_bias=position_bias,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        gnn_mask=gnn_mask,  # index of the gnn tokens
                        rel_mask=rel_mask,  # index of the rel tokens
                        gnn_triplets=gnn_triplets,  # list of triplets
                        memory_nodes=memory_nodes,  # node memory
                        current_reasoning_path=current_reasoning_path,  # current reasoning path
                        rels_ids = rels_ids,  # ids of the relations in the memory
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=extended_attention_mask,
                        position_bias=position_bias,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )


            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5GNNForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, args=None):
        super().__init__(config)

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.layer_with_gnn = args.layer_with_gnn
        encoder_config.n_rel = args.n_rel
        encoder_config.gnn_topk = args.gnn_topk
        encoder_config.gnn_embs_size = args.gnn_embs_size
        self.encoder = T5GNNStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True


    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        gnn_mask=None,  # index of the gnn tokens
        rel_mask=None,  # index of the rel tokens
        gnn_triplets=None,  # list of triplets
        memory_nodes=None,  # node memory
        current_reasoning_path: AllReasoningPath = None,
        rels_ids=None,  # ids of the relations in the memory,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration
        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                gnn_mask=gnn_mask,  # index of the gnn tokens
                rel_mask=rel_mask,  # index of the rel tokens
                gnn_triplets=gnn_triplets,  # list of triplets
                memory_nodes=memory_nodes,  # node memory
                current_reasoning_path = current_reasoning_path,
                rels_ids=rels_ids,  # ids of the relations in the memory
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            pdb.set_trace()
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        pdb.set_trace()
        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past





