# coding:utf-8
import sys
import torch
import numpy as np
import torch.nn as nn
from layer.tree import head_to_adj
from layer.RGAT import RGATEncoder
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RGATABSA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = ABSAEncoder(args)
        self.dropout = nn.Dropout(0.1)
        in_dim = args.bert_out_dim

        self.decoder = NoiseReduction(
            in_dim=in_dim,
            out_dim=args.num_class,
            dropout=self.dropout
        )

    def forward(self, inputs):
        outputs, sentence_noise = self.encoder(inputs)
        logits, sentence_logits, noise_logits =\
            self.decoder(outputs, sentence_noise)
        return \
            logits,\
            sentence_logits,\
            noise_logits


class ABSAEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dep_emb = (
            nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0)
            if args.dep_dim > 0
            else None
        )  # position emb
        self.encoder = DoubleEncoder(
            args,
            embeddings=self.dep_emb,
            use_dep=True
        )
        self.gate_map = nn.Linear(args.bert_out_dim * 2, args.bert_out_dim)

    def forward(self, inputs):
        (
            tok,
            asp,
            head,
            deprel,
            mask,
            l,
            bert_sequence,
            bert_segments_ids,
        ) = inputs  # unpack inputs
        maxlen = max(l.data)

        adj_lst, label_lst = [], []
        graph_generate = head_to_adj
        for idx in range(len(l)):
            adj_i, label_i = graph_generate(
                maxlen,
                head[idx],
                deprel[idx],
                l[idx],
                mask[idx],
                directed=self.args.direct,
                self_loop=self.args.loop,
            )
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))

        adj = np.concatenate(adj_lst, axis=0)  # [B, maxlen, maxlen]
        adj = torch.from_numpy(adj).cuda()

        labels = np.concatenate(label_lst, axis=0)  # [B, maxlen, maxlen]
        label_all = torch.from_numpy(labels).cuda()
        h = self.encoder(
            adj=adj,
            relation_matrix=label_all,
            inputs=inputs,
            lengths=l
        )

        graph_out, bert_out, inv_graph_out, \
            inv_bert_out, bert_pool_output, word_mask\
            = h[0], h[1], h[2], h[3], h[4], h[5]
        inv_outputs = self.merge(inv_graph_out, inv_bert_out, word_mask)

        merged_outputs = self.merge(graph_out, bert_out, word_mask)

        # 仅与是否使用cls的结果有关
        cat_outputs = merged_outputs

        return cat_outputs, inv_outputs

    def merge(self, graph_out, bert_out, word_mask):
        word_mask = (~word_mask).float()
        gate = torch.sigmoid(
            self.gate_map(
                torch.cat([graph_out, bert_out], -1)
            )
        )
        merged_outputs = gate*graph_out + (1-gate)*bert_out
        merged_outputs = merged_outputs * word_mask.unsqueeze(-1)
        merged_outputs = \
            merged_outputs.sum(1) / word_mask.sum(-1).unsqueeze(-1)
        return merged_outputs


class DoubleEncoder(nn.Module):
    def __init__(self, args, embeddings=None, use_dep=False):
        super(DoubleEncoder, self).__init__()
        self.args = args
        bert_config = BertConfig.from_pretrained(args.bert_from)
        bert_config.output_hidden_states = True
        bert_config.num_labels = 3
        bert = BertModel.from_pretrained(
            args.bert_from, config=bert_config
        )
        self.Sent_encoder = bert
        self.dense = nn.Linear(args.hidden_dim, args.bert_out_dim)
        self.in_drop = nn.Dropout(args.input_dropout)
        self.dep_emb = embeddings
        self.Graph_encoder = RGATEncoder(
            num_layers=args.num_layer,
            d_model=args.bert_out_dim,
            heads=4,
            d_ff=args.hidden_dim,
            dep_dim=self.args.dep_dim,
            att_drop=self.args.att_dropout,
            dropout=0.0,
            use_structure=True
        )
        if args.reset_pooling:
            self.reset_params(bert.pooler.dense)
        self.PTR = PtrEncoding(args.hidden_dim, args.input_dropout)

    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self, adj, inputs, lengths,
        relation_matrix=None
    ):
        (
            tok,
            asp,
            head,
            deprel,
            a_mask,
            l,
            bert_sequence,
            bert_segments_ids,
        ) = inputs  # unpack inputs

        if adj is not None:
            mask = adj.eq(0)
        else:
            mask = None
        # print('adj mask', mask, mask.size())
        if lengths is not None:
            key_padding_mask = sequence_mask(lengths)  # [B, seq_len]

        if relation_matrix is not None:
            dep_relation_embs = self.dep_emb(relation_matrix)
        else:
            dep_relation_embs = None

        bert_sequence = bert_sequence[:, 0:bert_segments_ids.size(1)]
        # input()
        bert_out, bert_pool_output, bert_all_out = self.Sent_encoder(
            bert_sequence, token_type_ids=bert_segments_ids
        )

        bert_out_origin = bert_out[:, 0:max(l), :]
        bert_out, graph_out = self.ptr_graph_encode(
            bert_out_origin, a_mask, mask,
            key_padding_mask,
            dep_relation_embs
        )
        inv_bert_out, inv_graph_out = self.ptr_graph_encode(
            bert_out_origin, (1-a_mask), mask,
            key_padding_mask,
            dep_relation_embs
        )

        return \
            graph_out, bert_out,\
            inv_graph_out, inv_bert_out,\
            bert_pool_output, key_padding_mask

    def ptr_graph_encode(
        self, bert_out, a_mask,
        mask,
        key_padding_mask,
        dep_relation_embs
    ):
        bert_out = self.PTR(bert_out, a_mask.long())
        bert_out = self.dense(bert_out)
        inp = bert_out  # [bsz, seq_len, H]
        graph_out = self.Graph_encoder(
            inp, mask=mask,
            src_key_padding_mask=key_padding_mask,
            structure=dep_relation_embs
        )               # [bsz, seq_len, H]
        return bert_out, graph_out


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(
        0, max_len, device=lengths.device
    ).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))


class NoiseReduction(nn.Module):
    '''
    init
        decoder -> decoder for the latent space
    '''
    def __init__(self, in_dim, out_dim, dropout):
        super(NoiseReduction, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, out_dim)
        )
        self.sentence_decoder = nn.Sequential(
            nn.Linear(2*in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, out_dim)
        )
        self.dropout = dropout

    def forward(self, outputs, sentence_noise):
        '''
        output the label for auxiliary dataset
        '''
        logits = self.decoder(self.dropout(outputs))
        sentence_logits = self.sentence_decoder(
            self.dropout(
                torch.cat([sentence_noise, outputs], dim=-1)
            )
        )
        noise_logits = self.decoder(self.dropout(sentence_noise))
        return logits, sentence_logits, noise_logits


class PtrEncoding(nn.Module):
    "Implement the PTR function."
    def __init__(self, d_model, dropout):
        super(PtrEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ptr only contain 0 and 1
        self.PTR = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

    def forward(self, x, ptr):
        '''
        x: [batch_size, seq_len, emb_size]
        ptr: [batch_size, seq_len] only has 0 and 1, long type
        '''
        x = x + self.PTR(ptr)
        return self.dropout(x)


class BERTBase(nn.Module):
    def __init__(self, args):
        super(BERTBase, self).__init__()
        bert_config = BertConfig.from_pretrained(args.bert_from)
        bert_config.output_hidden_states = True
        bert_config.num_labels = 3
        bert = BertModel.from_pretrained(
            args.bert_from, config=bert_config
        )
        self.args = args
        self.encoder = bert

    def forward(self, inputs):
        (
            tok,
            asp,
            head,
            deprel,
            a_mask,
            l,
            bert_sequence,
            bert_segments_ids,
        ) = inputs
        
        bert_sequence = bert_sequence[:, 0:bert_segments_ids.size(1)]
        # input()
        bert_out, bert_pool_output, bert_all_out = self.Sent_encoder(
            bert_sequence, token_type_ids=bert_segments_ids
        )
        word_mask = sequence_mask(l)  # [B, seq_len]

        merged_outputs = bert_out * word_mask.unsqueeze(-1)
        merged_outputs = \
            merged_outputs.sum(1) / word_mask.sum(-1).unsqueeze(-1)
        noise_out = bert_out * (1-word_mask).unsqueeze(-1)
        noise_out = \
            noise_out.sum(1) / word_mask.sum(-1).unsqueeze(-1)
        return merged_outputs, bert_pool_output, noise_out  # fit the output tuple format


class MultiBERT(nn.Module):
    def __init__(self, args):
        super(MultiBERT, self).__init__()
        self.args = args
        self.encoder = BERTBase
        self.dropout = nn.Dropout(0.5)
        self.asp_decoder = nn.Linear(args.hidden_dim, args.num_class)
        self.decoder = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, inputs):
        (
            tok,
            asp,
            head,
            deprel,
            a_mask,
            l,
            bert_sequence,
            bert_segments_ids,
        ) = inputs
        merged_outputs, bert_pool_output, noise = self.encoder(inputs)
        asp_logits = self.asp_decoder(self.dropout(merged_outputs))
        logits = self.decoder(self.dropout(bert_pool_output))
        return asp_logits, logits, 0  # fit the output tuple format


class MultiBERT_DPL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = BERTBase(args)
        self.dropout = nn.Dropout(0.1)
        in_dim = args.bert_out_dim

        self.decoder = NoiseReduction(
            in_dim=in_dim,
            out_dim=args.num_class,
            dropout=self.dropout
        )

    def forward(self, inputs):
        outputs, sentence, sentence_noise = self.encoder(inputs)
        logits, sentence_logits, noise_logits =\
            self.decoder(outputs, sentence_noise)
        return \
            logits,\
            sentence_logits,\
            noise_logits
