import math

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import BertModel, BertConfig

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.encoder import PositionalEncoding, TransformerEncoderLayer
from models.model_builder import Bert
from .fairseq_transformer import *
from fairseq import utils
from fairseq.models import (
    # FairseqEncoder,
    FairseqIncrementalDecoder
)
from fairseq.modules import (
    AdaptiveSoftmax,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    # DynamicConv1dTBC,
)
from jigsaw.tools import logical_not


class SentenceTransformer(nn.Module):
    # 我希望这个模型能以后作为摘要的模型，所以sent transformer encoder 之类的可能需要共享，之后 摘要在上面加transformer encoder 换classifier
    def __init__(self, args, device, checkpoint, sum_or_jigsaw):
        super(SentenceTransformer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.encoder = None
        args.encoder_embed_dim = self.bert.model.config.hidden_size
        args.decoder_embed_dim = self.bert.model.config.hidden_size
        self.emb_dim = self.bert.model.config.hidden_size
        self.encoder = SentTransformerEncoder(args)
        self.doc_symbol = args.doc_symbol if 'doc_symbol' in args else 0
        safe_max_pos = args.safe_max_pos
        if safe_max_pos > 512:
            my_pos_embeddings = nn.Embedding(safe_max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            # my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(safe_max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        bos_fc = args.bos_fc if 'bos_fc' in args else 0
        self.qr = args.qr if 'qr' in args else 'none'
        if bos_fc:
            self.bos_fc = Linear(self.emb_dim, self.emb_dim)
        else:
            self.bos_fc = None
        if args.output_fc:
            self.output_fc = Linear(self.emb_dim, self.emb_dim)
        else:
            self.output_fc = None
        self.sum_or_jigsaw = sum_or_jigsaw # 0 sum 1jigsaw
        self.decoder = SentTransformerDecoder(args)
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls, labels):
        # label : for sum is hot vector. for jigsaw is poss
        # label和idx 决定实时sent pos emb

        # 1. bert for token level representations
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        bsz, num_sent,dim = sents_vec.size()

        # 2. encoder
        if self.doc_symbol:
            doc = torch.zeros((bsz, 1,dim)).type_as(sents_vec)
            sents_vec = torch.cat((doc, sents_vec),dim=1)
            mask_cls = torch.cat((torch.ones(bsz, 1).type_as(mask_cls), mask_cls), dim=1)
        encoder_out = self.encoder(sents_vec, mask_cls)  # dict: encoder_out T x B x C; encoder_padding_mask B xT
        # prev_output_tokens bsz, num_sent, dim,
        # 无论是否teacher forcing 第一个bos 总得是自己的，我的做法是 w_bos * avg(encoder state) 作为bos

        # 3. decoder

        # 3.1 dynamic input embeddings, from idx to representations
        if self.doc_symbol:
            bos = encoder_out['encoder_out'][0, :, :]  # B C
            sent_reps = torch.transpose(encoder_out['encoder_out'][1:, :, :], 0, 1)  # B T C
            mask_cls = mask_cls[:, 1:]  # except doc symbol
        else:
            bos = torch.mean(encoder_out['encoder_out'], dim=0)  # B C
            sent_reps = torch.transpose(encoder_out['encoder_out'], 0, 1)  # B T C
        sent_reps = sent_reps * mask_cls[:, :, None].float()
        if self.bos_fc is not None:
            bos = F.tanh(self.bos_fc(bos))
        if self.sum_or_jigsaw == 0:  # 0 sum
            tgt_len = 3
            _, labels_id = torch.topk(labels, k=tgt_len)  # B, tgt_len
            labels_id, _ = torch.sort(labels_id)
            input_embeddings = sent_reps
            labels_rep = batched_index_select(input_embeddings, dim=1, inds=labels_id)  # B,tgt_len,C
            # shifted right and feed decoder
            prev_output_rep = torch.cat((bos.unsqueeze(1),  # B , tgt_len, C
                                         torch.narrow(labels_rep, dim=1, start=0, length=tgt_len-1)), dim=1)
            prev_output_tokens = torch.ones((bsz, tgt_len)).type_as(labels_id)  # no pad
        else:  # self.sum_or_jigsaw == 1:  # 1 jigsaw select 时还得考虑pad的问题，pad就不select
            tgt_len = num_sent
            input_embeddings = batched_index_select(sent_reps, dim=1, inds=labels)  # B , tgt_len, C
            # shifted right and feed decoder
            prev_output_rep = torch.cat(  # bsz , tgt_len, dim
                (bos.unsqueeze(1), torch.narrow(input_embeddings, dim=1, start=0, length=tgt_len - 1)), dim=1)
            prev_output_tokens = torch.narrow(torch.cat((torch.ones(bsz, 1).type_as(mask_cls), mask_cls), dim=1),
                                              dim=-1, start=0, length=tgt_len).type_as(labels)  # bool to long
        # 3.2 decoder
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens, prev_output_rep=prev_output_rep, encoder_out=encoder_out)
        dec_rep = decoder_out[0]  # B,tgt_len,C

        # 3.3 dynamic output layer from representations to idx
        # 主要是模仿output layer, 如果是bilinear, 只有一个w矩阵，给cls的还是decoder out 谁乘好像都是等价的
        if self.output_fc is not None:
            dec_rep = self.output_fc(dec_rep)
        scores = torch.bmm(dec_rep, input_embeddings.transpose(1, 2))  # B,tgt_len,T
        # 预测时再考虑mask 之前选择过的，训练时只能一次平行decode，或者逐行把分数和前面正交
        if self.qr in ['before_nosoftmax','before_softmax']:
            scores, _ = torch.qr(scores.transpose(1, 2))  # B,T,tgt_len,
            scores.transpose_(1, 2)  # B,tgt_len,T
            if self.qr == 'before_softmax':
                scores = F.softmax(scores,dim=-1)
        else:
            scores = F.softmax(scores,dim=-1)
            if self.qr == 'after_softmax':
                scores, _ = torch.qr(scores.transpose(1, 2))  # B,T,tgt_len,
                scores.transpose_(1, 2)  # B,tgt_len,T
        return scores, mask_cls

    def iterative_decoding(self):
        # 是拿前面分类选择去映射encoder out 得到表示 feed 还是拿前面的表示呢，觉得拿选择可能太过间接，相当于拿到词去重新映射了
        # 现在每个文档cls对应的表示类似一个绝对位置有无数个不同实时embedding 的embedding
        # 如果真的把句子表示和id的对应，每个文档一个emb, 那是否可以通过转置逆学回来了，或者就像是匹配一样，拿decoder的表示去匹配
        # 实际第i句话在encoder的表示，相当于把decoder第j个位置的输出拿到encoder 挨个算相似度
        # 这样和context selection那篇文章无比像了， 只是我说了encoder学到的句子表示当做实时sentence pos emb,
        # 缺点是没有反向attn这么炫酷的东西了，主要是decoder 递增反向attn，需要的运算也很大，可以放在ablation里试一下
        # 反向attn强调的是轮次，一轮选择两轮选择，每次都把所有的decode history 考虑了，现在是从decoder角度看一步两步
        # transformer 预测第i个词也没把i-1 前所有预测都拿来一起算啊, 而且用bos而非encoder state 平均来预测第一个词
        # 我之前想用inverse attn 是觉得rnn可以用state
        # 可以命名为static和dynamic pos emb(input output)  dynamic 是每个文档一个
        # 决定暂时放弃inverse attn ,考虑拿doc 符号作为第0个decode位置输入
        # 原来bos 也是考虑单词vocab静态采用， 现在文档位置动态，所以要么用doc符号作为输入，要么avg
        # 想办法在finetune时 保持mlm 比如用bert数据处理好后然后在roberta模型基础上finetune 一个textjigsac分类
        # 相比hibert每个句子都收到了监督信息 20200329 1616
        # 要做的后续事情很多，所以要先在这里试好再改到fairseq 试pretrain,主要是在roberta基础上保留mlm结果不会太差
        # 可以先用这个模型直接跑抽取式摘要
        # padding 是个大难题，考虑要不要预测eos,算了每个通过非0 就知道实际长度了，decode 是可以到哪停止，但是train 时w
        # decode batch内句子数最多的那个长度的话，后面的就不参与监督，其实也还可以，毕竟后面不影响前面
        # 现在打开手机也没用，那个104的空跑程序问题无解，只有期待今晚把程序跑上去
        # 问题是decoder的输入我这里用的是encoder的输出--正序版，但是attention的也是encoder 的输出，要不要额外几层encoder
        return

    def inference(self,src, segs, clss, mask_src, mask_cls):
        return


class SentTransformerEncoder(nn.Module):
    def __init__(self, args):
        super(SentTransformerEncoder, self).__init__()
        self.encoder_layers = args.encoder_layers
        if args.sent_pos_emb_enc:  # whether use pos emb for inter sentence encoding
            self.pos_emb = PositionalEncoding(args.dropout, args.encoder_embed_dim)
        else:
            self.pos_emb = None
        self.layers = nn.ModuleList([])
        self.layers.extend([
                TransformerEncoderLayer(args=args)
                for _ in range(args.encoder_layers)
            ])
        self.dropout = nn.Dropout(args.dropout)
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)

    def forward(self, top_vecs, mask):
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        x = top_vecs * mask[:, :, None].float()
        if self.pos_emb is not None:
            pos_emb = self.pos_emb.pe[:, :n_sents]
            x = x + pos_emb
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        encoder_padding_mask = logical_not(mask)
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)  # all_sents * max_tokens * dim
        if self.normalize:
            x = self.layer_norm(x)
        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


class SentTransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(self, args, no_encoder_attn=False, final_norm=True):
        super(SentTransformerDecoder, self).__init__()
        self.dropout = args.dropout
        embed_dim = args.decoder_embed_dim
        self.max_target_positions = args.max_target_positions
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx=0,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
                    TransformerDecoderLayer(args=args, no_encoder_attn=no_encoder_attn)
                    for i in range(args.decoder_layers)
                ])
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, prev_output_rep, encoder_out=None, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            prev_output_rep
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, dim)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens,prev_output_rep, encoder_out, incremental_state)
        # x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, prev_output_rep, encoder_out=None, incremental_state=None):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        # (batch, tgt_len)
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            # incre decoding 时就取最后一个 前面的都已经缓存了
            prev_output_tokens = prev_output_tokens[:, -1:]
            prev_output_rep = prev_output_rep[:,-1,:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        # x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = prev_output_rep
        # if self.project_in_dim is not None:
        #     x = self.project_in_dim(x)

        if positions is not None:
            # print('len x %s |len positions %s'%(x.size(),positions.size()))
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


# Batched index_select
def batched_index_select(t, dim, inds):
    #t bsz * num * dim
    # dim=1
    # bsz * k
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # bsz * k *dim
    return out
