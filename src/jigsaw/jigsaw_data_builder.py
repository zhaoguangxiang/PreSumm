import json
import os
import gc
import glob
import torch
import random
from os.path import join as pjoin
from multiprocess import Pool
from others.logging import logger
from others.tokenization import BertTokenizer
from prepro.data_builder import greedy_selection,BertData
import bisect
# import numpy as np


class JigsawData(BertData):
    # 没写完，关于数据shuffle 很多次，以及按顺序选择 python 可能支持的不好，考虑在data_loader 里用torch 实现就是，现场shuffl. 20200322 22:04
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'  # 这里是为了做abstrative的decoder
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def hot_to_id(self, hot_sent_labels):
        org_sent_labels = []
        for i, b in enumerate(hot_sent_labels):
            if b == 1:
                org_sent_labels.append(i)
        return org_sent_labels

    def tok_source(self, src):
        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        # cls sent sep cls sent sep cls sent sep cls sent sep  sep可能是来自bert 所以要用吧，document cls 和原来的可能不一样，或者就第一个cls
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)  # 这里已经是一维数组了
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]  # 算出在每个句子的长度以给每个位置加上相应的segment
        segments_ids = []  # 在jigsaw任务里不能输入这个 防止泄露
        # position_ids = []  # jigsaw 任务的目标
        for i, s in enumerate(segs):
            # position_ids.append(i)  # 2020/4/1 16:13  发现position_ids 在这里重新生成了正序的一个，是错的了
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]  # 在词级别的位置
        return src_txt, src_subtoken_idxs, segments_ids, cls_ids

    def preprocess_jigsaw(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False, times=1, unchange_prob=0.05):
        # input    source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, is_test=is_test
        # output   src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt
        datas = []
        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]  # idxs  src 中句子长度大于5的句号

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]  # src 是二维数组， 每句有最长限制，对最少句子数做了限制
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]

        sent_labels = sent_labels[:self.args.max_src_nsents] # 最多不会多于100句，真的有这么长吗？我统计下长度，然后决定，比如median max,mean,min,可能20或15就差不多了
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)
        # cls sent sep cls sent sep cls sent sep cls sent sep  sep可能是来自bert 所以要用吧，document cls 和原来的可能不一样，或者就第一个cls
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens) # 这里已经是一维数组了
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]  # 算出在每个句子的长度以给每个位置加上相应的segment
        segments_ids = []  # 在jigsaw任务里不能输入这个 防止泄露
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]  # 所有cls在词序列中的位置
        sent_labels = sent_labels[:len(cls_ids)]  # 0324 0958  1455 确认了这个没意义，因为一直一样

        max_pos = self.args.max_pos
        if max_pos != -1:  # 20200325 22:21因为后面涉及翻转，所以必须在翻转前但是句内截断后截断句子 2237 完成操作
            end_id = [src_subtoken_idxs[-1]]
            src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos - 1] + end_id
            segments_ids = segments_ids[:max_pos]
            max_sent_id = bisect.bisect_left(cls_ids, max_pos)
            sent_labels = sent_labels[:max_sent_id]
            cls_ids = cls_ids[:max_sent_id]
            last_len = max_pos - 1 - cls_ids[max_sent_id-1]
            src[max_sent_id-1] = src[max_sent_id-1][:last_len]

        position_ids = [i for i in range(len(sent_labels))] # jigsaw 任务的目标
        org_sent_labels = self.hot_to_id(sent_labels)

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        # 'src_s': src_subtoken_idxs_s, 'tgt_s': tgt_subtoken_idxs_s,
        # "src_sent_labels_s": sent_labels_s, "segs_s": segments_ids_s, 'clss_s': cls_ids_s,
        # "org_sent_labels_s": org_sent_labels, 'poss_s': poss_s,
        data_org = (src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt,  position_ids, org_sent_labels)
        datas.append(data_org)
        for i in range(times):
            # random.seed(i) # 每次设置seed后都重置了，所以这里不能设置seed
            if random.random() < unchange_prob:
                datas.append((src_subtoken_idxs, sent_labels, segments_ids, cls_ids, src_txt,  position_ids, org_sent_labels))
                continue
            shuffle_ratio = self.args.shuffle_ratio if 'shuffle_ratio' in self.args else 1.0
            if shuffle_ratio == 1:
                position_ids_s = position_ids.copy()
                random.shuffle(position_ids_s) # 不可改变原list 就这么写
            else:
                position_ids_c = position_ids.copy()
                shuffle_len = int(len(position_ids)*shuffle_ratio+0.5)
                position_ids_cs = position_ids_c[-shuffle_len:]
                random.shuffle(position_ids_cs)
                position_ids_s = position_ids[:-shuffle_len] + position_ids_cs
            sent_labels_s = [sent_labels[pos] for pos in position_ids_s]
            # 上面是关于句子级的一维数组或者数组的数组, 直接寻址解决,下面直接抄上面original的代码，解决需要拼接以及加特殊字符的情况
            src_s = [src[pos] for pos in position_ids_s]
            src_txt_s, src_subtoken_idxs_s, segments_ids_s,  cls_ids_s = self.tok_source(src_s)
            # 这里的src text 和原本的不一样了，原来是只要长度大于5，这里有更多限制的再shuffle的，都是list,item句子
            sent_labels_s = sent_labels_s[:len(cls_ids_s)]  # useless
            org_sent_labels_s = self.hot_to_id(sent_labels_s)
            datas.append((src_subtoken_idxs_s, sent_labels_s, segments_ids_s, cls_ids_s, src_txt_s, position_ids_s,
                          org_sent_labels_s))
        return datas


def format_to_jigsaw(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_jigsaw, a_lst):
            pass

        pool.close()
        pool.join()


def _format_to_jigsaw(params):  # 比bert data 的preprocess 就多了读存数据和lower以及判断是否为空的的utils
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = JigsawData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))  # 一个jobs 是一个shard 现在膨胀了times 倍
    if args.sample_near:
        datasets = []
        for d in jobs:
            source, tgt = d['src'], d['tgt']
            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
            if (args.lower):
                source = [' '.join(s).lower().split() for s in source]
                tgt = [' '.join(s).lower().split() for s in tgt]
            b_data = bert.preprocess_jigsaw(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                     is_test=is_test, times=args.times,unchange_prob=args.unchange_prob)
            if (b_data is None):
                continue
            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, poss, org_sent_labels = b_data[0]
            for i in range(0, args.times):
                src_subtoken_idxs_s, sent_labels_s, segments_ids_s, cls_ids_s, src_txt_s, poss_s, org_sent_labels_s = b_data[i+1]
                if args.keep_orgdata:
                    b_data_dict = {'src_s': src_subtoken_idxs_s, "tgt": tgt_subtoken_idxs,
                                    "segs_s": segments_ids_s, 'clss_s': cls_ids_s, "src_sent_labels_s": sent_labels_s,
                                   "org_sent_labels_s": org_sent_labels_s, 'poss_s': poss_s,

                                   'src_txt_s': src_txt_s, "tgt_txt": tgt_txt,  'src_txt': src_txt,

                                   "src": src_subtoken_idxs,
                                   "segs": segments_ids, 'clss': cls_ids, "src_sent_labels": sent_labels,
                                   "org_sent_labels": org_sent_labels, 'poss': poss, }
                else:
                    b_data_dict = {
                        'src_s': src_subtoken_idxs_s, "tgt": tgt_subtoken_idxs,
                         "segs_s": segments_ids_s, 'clss_s': cls_ids_s, "src_sent_labels_s": sent_labels_s,
                        "org_sent_labels_s": org_sent_labels_s, 'poss_s': poss_s,

                        'src_txt_s': src_txt_s,  "tgt_txt": tgt_txt, 'src_tex': src_txt,
                    }
                datasets.append(b_data_dict)
        logger.info('Processed instances %d' % len(datasets))
        logger.info('Saving to %s' % save_file)
        torch.save(datasets, save_file)
        datasets=[]
    else:
        # datasets = []
        dataset_list = [ [] for i in range(args.times)]
        for d in jobs:
            source, tgt = d['src'], d['tgt']
            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)

            if (args.lower):
                source = [' '.join(s).lower().split() for s in source]
                tgt = [' '.join(s).lower().split() for s in tgt]
            b_data = bert.preprocess_jigsaw(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                            is_test=is_test, times=args.times,unchange_prob =args.unchange_prob)
            if (b_data is None):
                continue
            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, poss, org_sent_labels = b_data[0]
            for i in range(args.times):
                src_subtoken_idxs_s, sent_labels_s,  segments_ids_s, cls_ids_s, src_txt_s, poss_s, org_sent_labels_s = b_data[i+1]
                if args.keep_orgdata:
                    b_data_dict = {'src_s': src_subtoken_idxs_s, "tgt": tgt_subtoken_idxs,
                                    "segs_s": segments_ids_s, 'clss_s': cls_ids_s, "src_sent_labels_s": sent_labels_s,
                                   "org_sent_labels_s": org_sent_labels_s, 'poss_s': poss_s,

                                   'src_txt_s': src_txt_s, "tgt_txt": tgt_txt,  'src_txt': src_txt,

                                   "src": src_subtoken_idxs,
                                   "segs": segments_ids, 'clss': cls_ids, "src_sent_labels": sent_labels,
                                   "org_sent_labels": org_sent_labels, 'poss': poss, }
                else:
                    b_data_dict = {
                        'src_s': src_subtoken_idxs_s, "tgt": tgt_subtoken_idxs,
                         "segs_s": segments_ids_s, 'clss_s': cls_ids_s, "src_sent_labels_s": sent_labels_s,
                        "org_sent_labels_s": org_sent_labels_s, 'poss_s': poss_s,

                        'src_txt_s': src_txt_s,  "tgt_txt": tgt_txt, 'src_tex': src_txt,
                    }
                dataset_list[i].append(b_data_dict)
        for j in range(args.times):
            logger.info('Processed instances %d' % len(dataset_list[j]))
            # Train 从0到143， valid 从0到6， test 从0到5
            # cnndm_sample.train.0.bert.pt cnndm.valid.6.bert.pt
            filename = save_file.split('/')[-1]
            save_file_list = []
            save_file_split = filename.split('.')
            save_file_list.extend(save_file_split[:2])
            # ../ jigsaw_data / cnndm.train.93.bert.pt
            if save_file_split[0] == 'cnndm_sample':
                total = 1
            else:
                if save_file_split[1] == 'train':
                    total = 144
                elif save_file_split[1] == 'valid':
                    total = 7
                else:
                    total = 6
            save_file_list.append(str(total*j + int(save_file_split[2])))
            save_file_list.extend(save_file_split[3:])
            final_save_file = args.save_path + os.sep + '.'.join(save_file_list)
            logger.info('Saving to %s' % final_save_file)
            torch.save(dataset_list[j], final_save_file)
        datasets_list = []
    gc.collect()