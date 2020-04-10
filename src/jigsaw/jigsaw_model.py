import math

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import BertModel, BertConfig

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.encoder import PositionalEncoding, TransformerEncoderLayer
from models.model_builder import Bert


class Jigsaw(nn.Module):
    # 我希望这个模型能以后作为摘要的模型，所以sent transformer encoder 之类的可能需要共享，之后 摘要在上面加transformer encoder 换classifier
    def __init__(self, args, device, checkpoint):
        super(Jigsaw, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.basic_encoder = None
        self.high_encoder = None
        if args.jigsaw_basic_encoder_layers > 0:
            self.basic_encoder = SentTransformerEncoder(self.bert.model.config.hidden_size, args.jigsaw_ff_size, args.jigsaw_heads, args.jigsaw_dropout, args.jigsaw_basic_encoder_layers, args)
        if args.jigsaw_high_encoder_layers > 0:
            self.high_encoder = SentTransformerEncoder(self.bert.model.config.hidden_size, args.jigsaw_ff_size,
                                                           args.jigsaw_heads, args.jigsaw_dropout,
                                                           args.jigsaw_high_encoder_layers, args)
            if args.jigsaw_basic_encoder_layers > 0:
                self.high_encoder.pos_emb = self.basic_encoder.pos_emb

        if args.encoder == 'baseline':
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
        self.jigsaw_classifier = OutputLayer(self.bert.model.config.hidden_size, args=args, output_type='multi')

        # if args.max_pos>512:
        safe_max_pos = args.safe_max_pos
        if safe_max_pos > 512:
            my_pos_embeddings = nn.Embedding(safe_max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            # my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(safe_max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            ext_layers = [self.jigsaw_classifier]
            if self.basic_encoder is not None:
                ext_layers.append(self.basic_encoder)
            if self.high_encoder is not None:
                ext_layers.append(self.high_encoder)
            for ext_layer in ext_layers:
                if args.param_init != 0.0:
                    for p in ext_layer.parameters():
                        p.data.uniform_(-args.param_init, args.param_init)
                if args.param_init_glorot:
                    for p in ext_layer.parameters():
                        if p.dim() > 1:
                            xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        if self.basic_encoder is not None:
            sents_vec = self.basic_encoder(sents_vec, mask_cls)
        if self.high_encoder is not None:
            sents_vec = self.high_encoder(sents_vec, mask_cls)
        sent_scores = self.jigsaw_classifier(sents_vec, mask_cls)  # bsz, n_sent, max_n_sent
        return sent_scores


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
        sent_scores = self.fc(x)
        sent_scores.masked_fill_(mask_cls.unsqueeze(-1).repeat(1,1,sent_scores.size()[-1]), -1e4)
        sent_scores = self.output_act(sent_scores)   # bsz, n_sent, max_sent
        return sent_scores


