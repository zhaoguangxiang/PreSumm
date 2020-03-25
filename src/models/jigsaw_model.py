import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.encoder import PositionalEncoding, TransformerEncoderLayer

class SentTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0, args=None):
        super(SentTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        sent_pos_emb = args.sent_pos_emb
        if sent_pos_emb:
            self.pos_emb = PositionalEncoding(dropout, d_model)
        else:
            self.pos_emb = None
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        x = top_vecs * mask[:, :, None].float()
        if self.pos_emb is not None:
            pos_emb = self.pos_emb.pe[:, :n_sents]
            x = x + pos_emb
        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, torch.logical_not(mask))  # all_sents * max_tokens * dim
        x = self.layer_norm(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, hidden_size, args=None,output_type='multi'):
        super(OutputLayer, self).__init__()
        classifer_dim = args.max_src_nsents if output_type == 'multi' else 1  # output_type in [binary, multi]
        if args.output_fc_type == 'linear':
            self.fc = nn.Linear(hidden_size, args.classifer_dim)
        elif args.output_fc_type == 'nonlinear':
            self.fc = nn.Sequential(nn.Linear(hidden_size,hidden_size), nn.ReLU(), nn.Linear(hidden_size, classifer_dim))
        if output_type == 'multi':
            self.output_act = nn.Softmax()
        elif output_type == 'binary':
            self.output_act = nn.Sigmoid()

    def forward(self, x, mask_cls):
        sent_scores = self.output_act(self.fc(x)) * mask_cls[:, :, None].float()  # bsz, n_sent, max_sent
        return sent_scores