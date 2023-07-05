import math
import pdb
import random
from typing import Optional, Union, Tuple, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import BartConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right, BartPretrainedModel, _expand_mask, \
    BartEncoderLayer, BartLearnedPositionalEmbedding, BartAttention, BartDecoder

from tools import AllReasoningPath, extract_all_relations_for_a_node, extract_values_from_tensor, find_triplets


class CustomGNNLayer(torch.nn.Module):

    def __init__(self,
                 n_rel,  # Numero di tutte le possibili relazioni
                 embs_size,
                 model_size,
                 topk=2,
                 reprojection_activation='tanh',
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
        if reprojection_activation == 'tanh':
            not_linear_func = torch.nn.Tanh()
        if reprojection_activation == 'relu':
            not_linear_func = torch.nn.ReLU()
        if reprojection_activation == 'sigmoid':
            not_linear_func = torch.nn.Sigmoid()
        if reprojection_activation == 'elu':
            not_linear_func = torch.nn.ELU()
        if reprojection_activation == 'leaky_relu':
            not_linear_func = torch.nn.LeakyReLU()
        if reprojection_activation == 'selu':
            not_linear_func = torch.nn.SELU()

        self.query_reprj = torch.nn.Sequential(torch.nn.Linear(self.model_size, self.model_size), not_linear_func)
        self.nodes_reprj = torch.nn.Sequential(torch.nn.Linear(self.embs_size, self.model_size), not_linear_func)

        # Now the reprojection layer to inject graph knowledge into the GNN-TOK
        self.gnn_reprj = torch.nn.Sequential(torch.nn.Linear(self.embs_size, self.model_size), not_linear_func)

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
        # groups_stacked_tmp has the shape of (topk, max_grou_size, embs dim)
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

        weighted_attention = weighted_attention * mask[:,:, None].float()
        #print("wa", weighted_attention.shape)

        return weighted_attention, groups_stacked_tmp

    def forward(self,
                hidden_states,  #
                gnn_mask=None,  #
                rel_mask=None,  # index of the rel tokens
                gnn_triplets=None,  # list of triplets
                memory_embs=None,  # embeddings of the nodes in the memory
                current_reasoning_path: AllReasoningPath = None,
                rels_ids=None,  # ids of the relations in the memory
                ):
        #pdb.set_trace()
        """
        This layer is the core of the GNN-TOK. It takes as input the hidden states of the sentence and the graph and decide in which direction walk in the graph in order to
        get novel information. It returns the new hidden states of the sentence and the new reasoning path.

        :param hidden_states: embeddings of all the tokens in the sentence shape [bs, seq_len, emb_size] where bs=1
        :param gnn_mask: index of the gnn tokens. Binary vector that indicates (1) where are the <gnn_tok> tokens. Shape [bs, seq_len]
        :param rel_mask: index of the rel tokens. Binary vector that indicates (1) where are the <rel_tok> tokens. Shape [bs, seq_len]
        :param gnn_triplets: list of tripletes of strings (start, rel, end). It is the local graph of depth N.
        :param memory_embs: it is a dictionary containing where the k is the node name and the value its embeddings generated with a sentence transformer.
        :param current_reasoning_path: It is the object that contains the reasoning path.
        :param rels_ids: a dictionary of {relation:ids}. One id for each relation we have.
        :return:
        """
        #pdb.set_trace()
        try:
            assert hidden_states.shape[0] == 1, "The batch size must be 1"
            assert len(hidden_states.shape) == 3, "The hidden states must be 3 dimensional"
            assert hidden_states[gnn_mask.bool()].shape[0] > 0, "The GNN mask must be not empty"
            assert hidden_states[rel_mask.bool()].shape[0] > 0, "The REL mask must be not empty"
            assert memory_embs is not None, "The memory embeddings must be not None"
        except AssertionError as e:
            print(e)
            print("hidden_states.shape", hidden_states.shape)
            print("Selected hidden states by gnn", hidden_states[gnn_mask.bool()].shape)
            print("Selected hidden states by rel", hidden_states[rel_mask.bool()].shape)
            print("gnn_mask.shape", gnn_mask.shape)
            print("rel_mask.shape", rel_mask.shape)
            print("memory_embs.shape", memory_embs.shape)
            print("current_reasoning_path", current_reasoning_path.get_all_reasoning_path())
            exit()
        #pdb.set_trace()
        # I generate the probability over all the relations
        rel_prob = self.classification_head(hidden_states[rel_mask.bool()])
        # rel_prob shape (batch_size=1, num_rels)

        current_nodes = current_reasoning_path.get_current_nodes()
        # now current nodes is a dict where k arep the root nodes and v are a list of topk elements representing the current node

        # Now I extrac the queries which are the hidden states of the gnn tokens.
        # queries shape (num_gnn_tokens, emb_size)
        queries = hidden_states[gnn_mask.bool()]
        output = list()
        for query, probs, root_word in zip(queries, rel_prob, current_nodes.keys()):
            # I select the last nodes in the reasoning path generated by the root_node
            # the number of last_nodes is topk
            last_nodes = [n for n, _ in current_nodes[root_word]]

            # I extract their relations creating a list of tuple (node, [rels])
            rels_of_last_nodes = [(k, extract_all_relations_for_a_node(k, gnn_triplets)) for k in last_nodes]

            # I turn relations into ids using the rels_ids dictionary
            # It is a list of lists of ids. One list for each last node.
            rels_of_last_nodes_ids = [[rels_ids[r] for r in v] for k, v in rels_of_last_nodes]

            #pdb.set_trace()
            # I turn relation into probability. The probs contains the probability over all the possibile relations generated by the model.
            # it is a tensor of shape [n_rels].
            # The extract_values_from_tensor function extracts the values of the tensor in the positions specified by the ids.
            # In this way I can extract the associated probabilities to the relations of the last nodes.
            rels_of_last_nodes_prob = extract_values_from_tensor(probs, rels_of_last_nodes_ids)

            # Now I create the groups. A group is a set of triplets that has the same start, and rel.
            # I create a set of triplets (group) for each last node for each relation.
            # group_nodes contains only the end nodes in a structure equal to the rels_of_last_nodes_prob.
            # in this way for each prob in rels_of_last_nodes_prob corresponds a node in groups_nodes
            groups_nodes = list()
            selected_nodes = list()
            all_scores = list()

            # all the following variables I iterate over have the first size equal to the number of last nodes (topk)
            # so we can say: for each last node I select the relations in words, the associated probabilities, the associated ids
            for (k, rels), probs, ids in zip(rels_of_last_nodes, rels_of_last_nodes_prob, rels_of_last_nodes_ids):
                # groups_nodes contains a list for each pair (last_node, rel) containing all the end nodes of the pair.
                groups_nodes.append([[n for _, _, n in find_triplets(gnn_triplets, start=k, rel=rel)] for rel in rels])

            # now it computes the weighted contribution of all the ends nodes.
            # They are weighted by the relation weights and the score computed with an attention mechanism
            # Like before all the vaiable have the first size equal to the number of last nodes (topk) so we iterate for each last_word.
            for end_nods, probs, rels, cur_node, k in zip(groups_nodes, rels_of_last_nodes_prob, rels_of_last_nodes, last_nodes, range(self.topk)):
                # end_nods contains all the end nodes that have cur_node as start node
                # particularly end_nods is a list of list of end nodes. One list for each rel of cur_node.
                node_embs = list()
                for ns in end_nods:
                    # ns contains is the list of end nodes associated to cur_node and a rel.
                    # turn them into embedding using the memory matrix
                    node_embs.append(torch.stack([torch.tensor(memory_embs[n]) for n in ns]))
                # compute the scores associated to each embedding
                # the function perform a self-attention between the query and the node_embs extracted before.
                # the node_embs are of shape [n_rels, n_end_nodes, emb_dim]
                # the query is the embedding of the <gnn_token> tokens of the root_word of the cur_node path.
                # score contains the scores of each end_node in each group
                # embs is a padded tensor of shape [n_groups, n_end_nodes, emb_dim]
                scores, embs = self.calculate_scores(query.view(1, -1), k_nodes=node_embs,
                                                     probabilities=probs.view(1, -1))

                # now I have to weight the embeddings with the scores
                # first I have to reshape the scores and the embsp
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
                node_rel_list = [(r, n) for r, ns in zip(rels[1], end_nods) for n in ns]

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
        #pdb.set_trace()

        # Get the indices of ones in gnn_mask
        ones_indices = torch.nonzero(gnn_mask == 1)
        for t, ids in zip(output, ones_indices):
            outs = self.gnn_reprj(t)
            t_exp = torch.zeros_like(hidden_states)
            t_exp[0][ids[1]] = outs
            hidden_states += t_exp

        return hidden_states, current_reasoning_path



class BartGNNForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig, args=None):
        super().__init__(config)

        config.layer_with_gnn = args.layer_with_gnn
        config.n_rel = args.n_rel
        config.gnn_topk = args.gnn_topk
        config.gnn_embs_size = args.gnn_embs_size
        config.reprojection_activation = args.reprojection_activation

        self.model = BartGNNModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_and_clean_reasoning_path(self):
        return self.model.get_and_clean_reasoning_path()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
        memory_embs=None,  # embeddings of the nodes in the memory
        current_reasoning_path: AllReasoningPath = None,
        rels_ids=None,  # ids of the relations in the memory,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gnn_mask=gnn_mask,  # index of the gnn tokens
            rel_mask=rel_mask,  # index of the rel tokens
            gnn_triplets=gnn_triplets,  # list of triplets
            memory_embs=memory_embs,  # embeddings of the nodes in the memory
            current_reasoning_path=current_reasoning_path,
            rels_ids=rels_ids,  # ids of the relations in the memory
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)


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


class BartGNNModel(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartGNNEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        gnn_mask=None,  # index of the gnn tokens
        rel_mask=None,  # index of the rel tokens
        gnn_triplets=None,  # list of triplets
        memory_embs=None,  # embeddings of the nodes in the memory
        current_reasoning_path: AllReasoningPath = None,
        rels_ids=None,  # ids of the relations in the memory
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                gnn_mask=gnn_mask,  # index of the gnn tokens
                rel_mask=rel_mask,  # index of the rel tokens
                gnn_triplets=gnn_triplets,  # list of triplets
                memory_embs=memory_embs,  # embeddings of the nodes in the memory
                current_reasoning_path=current_reasoning_path,  # current reasoning path
                rels_ids=rels_ids,  # ids of the relations in the memory
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_and_clean_reasoning_path(self):
        return self.encoder.get_and_clean_reasoning_path()


class BartGNNEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )

        self.current_reasoning_path = None

        layers = list()
        for i in range(config.encoder_layers):
            if i in config.layer_with_gnn:
                print('Altering layer {} with GNN'.format(i))
                layers.append(BartGNNEncoderLayer(config))
            else:
                layers.append(BartEncoderLayer(config))

        self.layers = nn.ModuleList(layers)
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        gnn_mask=None,  # index of the gnn tokens
        rel_mask=None,  # index of the rel tokens
        gnn_triplets=None,  # list of triplets
        memory_embs=None,  # embeddings of the nodes in the memory
        current_reasoning_path: AllReasoningPath = None,
        rels_ids=None,  # ids of the relations in the memory
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    if isinstance(encoder_layer, BartEncoderLayer):
                        layer_outputs = checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                        )
                    else:
                        layer_outputs = checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                            gnn_mask=gnn_mask,  # index of the gnn tokens
                            rel_mask=rel_mask,  # index of the rel tokens
                            gnn_triplets=gnn_triplets,  # list of triplets
                            memory_embs=memory_embs,  # embeddings of the nodes in the memory
                            current_reasoning_path=current_reasoning_path,  # current reasoning path
                            rels_ids=rels_ids,  # ids of the relations in the memory
                        )

                else:
                    if isinstance(encoder_layer, BartGNNEncoderLayer):
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                            gnn_mask=gnn_mask,  # index of the gnn tokens
                            rel_mask=rel_mask,  # index of the rel tokens
                            gnn_triplets=gnn_triplets,  # list of triplets
                            memory_embs=memory_embs,  # embeddings of the nodes in the memory
                            current_reasoning_path=current_reasoning_path,  # current reasoning path
                            rels_ids=rels_ids,  # ids of the relations in the memory
                        )
                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                        )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        self.current_reasoning_path = current_reasoning_path

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    def get_and_clean_reasoning_path(self):
        reasoning_path = self.current_reasoning_path
        self.current_reasoning_path = None
        return reasoning_path


class BartGNNEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.layer = CustomGNNLayer(
            n_rel=config.n_rel, embs_size=config.gnn_embs_size, model_size=config.d_model, topk=config.gnn_topk,
            reprojection_activation=config.reprojection_activation
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        gnn_mask=None,  # index of the gnn tokens
        rel_mask=None,  # index of the rel tokens
        gnn_triplets=None,  # list of triplets
        memory_embs=None,  # embeddings of the nodes in the memory
        current_reasoning_path: AllReasoningPath = None,
        rels_ids=None,  # ids of the relations in the memory
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Apply GNN layer
        hidden_states, current_reasoning_path = self.layer(
            hidden_states,  # embeddings of all the tokens in the sentence
            gnn_mask,  # index of the gnn tokens
            rel_mask,  # index of the rel tokens
            gnn_triplets,  # list of triplets
            memory_embs,  # embeddings of the nodes in the memory
            current_reasoning_path,
            rels_ids,  # ids of the relations in the memory
            )


        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs



