#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext
from jigsaw.jigsaw_train import train_jigsaw, validate_jigsaw, test_jigsaw
model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs', 'jigsaw'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    # params for jigsaw
    # max_pos 我是按句子来的，这个可能也超512了， 感觉还是hibert 句内pos emb 靠谱
    parser.add_argument("-jigsaw", default= 'jigsaw_dec', type=str,choices=['jigsaw_lab', 'jigsaw_dec'])
    parser.add_argument("-keep_orgdata", default=0, type=int)  # be consitent with preprocess
    parser.add_argument('-max_src_nsents', default=20, type=int)  # be consitent with preprocess
    parser.add_argument("-sent_pos_emb", default=0, type=int,help='for org seq lab jigsaw')  # used in encoder/jigsaw transformer encoder
    parser.add_argument("-safe_max_pos", default=1024, type=int)

    # for the org jigsaw model
    parser.add_argument('-jigsaw_basic_encoder_layers', default=0, type=int, help='share for downstream tasks')
    parser.add_argument('-jigsaw_high_encoder_layers', default=0, type=int,
                        help='')
    parser.add_argument("-jigsaw_dropout", default=0.2, type=float)
    parser.add_argument("-jigsaw_hidden_size", default=768, type=int)
    parser.add_argument("-jigsaw_heads", default=8, type=int)
    parser.add_argument("-jigsaw_ff_size", default=2048, type=int)
    parser.add_argument("-output_fc_type", default='linear', type=str, choices=['linear', 'nonlinear'])

    # for jigsaw transformer
    parser.add_argument("-sent_pos_emb_enc", default=0, type=int,help='0 for jisaw 1 for sum')
    parser.add_argument("-dropout", default=0.2, type=float)
    parser.add_argument("-attention_dropout", default=0.0, type=float)
    parser.add_argument("-encoder_embed_dim", default=768, type=int)
    parser.add_argument("-encoder_attention_heads", default=8, type=int)
    parser.add_argument("-encoder_normalize_before", default=0, type=int)
    parser.add_argument("-encoder_ffn_embed_dim", default=2048, type=int)
    parser.add_argument("-decoder_embed_dim", default=768, type=int)
    parser.add_argument("-decoder_attention_heads", default=8, type=int)
    parser.add_argument("-decoder_normalize_before", default=0, type=int)
    parser.add_argument("-decoder_ffn_embed_dim", default=2048, type=int)
    parser.add_argument("-no_token_positional_embeddings", default=0, type=int)
    parser.add_argument("-encoder_layers", default=2, type=int)
    parser.add_argument("-decoder_layers", default=2, type=int)
    parser.add_argument("-max_target_positions", default=20, type=int)
    parser.add_argument("-decoder_learned_pos", default=1, type=int)
    parser.add_argument("-bos_fc", default=1, type=int)
    parser.add_argument("-doc_symbol", default=0, type=int)
    parser.add_argument("-output_fc", default=1, type=int)
    parser.add_argument("-qr", default='none', type=str,
                        choices=['before_nosoftmax', 'before_softmax','none','after_softmax'])

    parser.add_argument("-ext_sum_dec", default=0, type=int)

    parser.add_argument("-fp16", default=0, type=int)




    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-print_every", default=0, type=int)
    parser.add_argument('-weight_up', type=float, default=1)

    parser.add_argument("-reset_optim", default=0, type=int)
    parser.add_argument("-acc_reporter", default=0, type=int)
    parser.add_argument('-shuffle_ratio', type=float, default=1.0)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    # args.gpu_ranks = [3,4]
    # args.world_size = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args, device_id)
        elif (args.mode == 'validate'):
            validate_abs(args, device_id)
        elif (args.mode == 'lead'):
            baseline(args, cal_lead=True)
        elif (args.mode == 'oracle'):
            baseline(args, cal_oracle=True)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)

    elif (args.task == 'ext'):
        if (args.mode == 'train'):
            train_ext(args, device_id)
        elif (args.mode == 'validate'):
            validate_ext(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)
    elif args.task == 'jigsaw':
        if args.mode == 'train':
            train_jigsaw(args, device_id)
        elif args.mode == 'validate':
            validate_jigsaw(args, device_id)
        elif args.mode == 'test':
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_jigsaw(args, device_id, cp, step)