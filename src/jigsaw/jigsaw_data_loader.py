import bisect
import gc
import glob
import random

import torch

from others.logging import logger
from jigsaw.tools import logical_not


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def batch_pre(self, pre_src,pre_segs,pre_clss, pre_src_sent_labels, org_sent_labels, poss, device, postfix):
        src = torch.tensor(self._pad(pre_src, 0))
        segs = torch.tensor(self._pad(pre_segs, 0))
        mask_src = logical_not(src == 0)
        clss = torch.tensor(self._pad(pre_clss, -1))
        src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
        mask_cls = logical_not(clss == -1)
        clss[clss == -1] = 0
        setattr(self, 'clss' + postfix, clss.to(device))
        setattr(self, 'mask_cls' + postfix, mask_cls.to(device))
        setattr(self, 'src_sent_labels' + postfix, src_sent_labels.to(device))
        setattr(self, 'src' + postfix, src.to(device))
        setattr(self, 'segs' + postfix, segs.to(device))
        setattr(self, 'mask_src' + postfix, mask_src.to(device))
        # 下面都是要预测的给他pad -1, 意思是看到-1 就停止算loss, 不用计算mask ,mask 是作为输入时才要的
        org_sent_labels = torch.tensor(self._pad(org_sent_labels, -1))
        setattr(self, 'org_sent_labels' + postfix, org_sent_labels.to(device))
        poss = torch.tensor(self._pad(poss, -1))  # 和sent albels 不一样，不能用0pad, 因为0也是一个意义值，用-1合适
        setattr(self, 'poss' + postfix, poss.to(device))

        assert torch.equal(logical_not(poss == -1), mask_cls)  # 2020/3/30 18:18 如果这关过了就很开心了
        # dec_poss = torch.tensor(self._pad(dec_poss, -1))
        # setattr(self, 'poss' + postfix, poss.to(device))

    def __init__(self, data=None, device=None, is_test=False, keep=0):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            if keep:
                data0 = [x[:-6] for x in data]
                data1 = [x[-6:] for x in data]
            else:
                data0 = data
            # (src_s, tgt, segs_s, clss_s, src_sent_labels_s, org_sent_labels_s, poss_s, src_txt_s, tgt_txt,
            #  src_txt), (src, segs, clss, src_sent_labels, org_sent_labels, poss)
            pre_src_s = [x[0] for x in data0]
            pre_segs_s = [x[2] for x in data0]
            pre_clss_s = [x[3] for x in data0]
            pre_src_sent_labels_s = [x[4] for x in data0]
            org_sent_labels_s = [x[5] for x in data0]
            # print('data0 item len ', len(data0[-1]),len(data0[-2]), ' | data1 item len: ', len(data1[-1]),len(data1[-2]))
            poss_s = [x[6] for x in data0]
            self.batch_pre(pre_src=pre_src_s, pre_segs=pre_segs_s, pre_clss=pre_clss_s,
                           pre_src_sent_labels=pre_src_sent_labels_s, org_sent_labels=org_sent_labels_s, poss=poss_s,
                           device=device, postfix='_s')
            if keep:
                pre_src = [x[0] for x in data1]
                pre_segs = [x[1] for x in data1]
                pre_clss = [x[2] for x in data1]
                pre_src_sent_labels = [x[3] for x in data1]
                org_sent_labels = [x[4] for x in data1]
                poss = [x[5] for x in data1]
                self.batch_pre(pre_src=pre_src, pre_segs=pre_segs, pre_clss=pre_clss,
                               pre_src_sent_labels=pre_src_sent_labels, org_sent_labels=org_sent_labels, poss=poss,
                               device=device,  postfix='')
            pre_tgt = [x[1] for x in data0]
            tgt = torch.tensor(self._pad(pre_tgt, 0))
            mask_tgt = logical_not(tgt == 0)
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))

            if (is_test):
                src_str_s = [x[-3] for x in data0]
                setattr(self, 'src_str_s', src_str_s)
                src_str = [x[-2] for x in data0]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data0]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size

def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if (len(new) == 4):
        pass
    src = new[0]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if (self.args.task == 'abs'):
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        # 这里主要作用是把文档中的句子数按max_pos 截断，max_pos 是512，我在预处理时截断25句，这里就不截断了，因为不能我句子都打乱好了，你这里再和我说截断，那不是很多序号靠前的句子没了
        # b_data_dict = {'src_s': src_subtoken_idxs_s, "tgt": tgt_subtoken_idxs,
        #                    "segs_s": segments_ids_s, 'clss_s': cls_ids_s, "src_sent_labels_s": sent_labels_s,
        #                    "org_sent_labels_s": org_sent_labels_s, 'poss_s': poss_s,
        #
        #                    'src_txt_s': src_txt_s, "tgt_txt": tgt_txt, 'src_txt': src_txt,
        #
        #                    "src": src_subtoken_idxs,
        #                    "segs": segments_ids, 'clss': cls_ids, "src_sent_labels": sent_labels,
        #                    "org_sent_labels": org_sent_labels, 'poss': poss, }
        src_s = ex['src_s']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1]+[2]
        src_sent_labels_s = ex['src_sent_labels_s']
        segs_s = ex['segs_s']
        if(not self.args.use_interval):
            segs_s = [0]*len(segs_s)
        clss_s = ex['clss_s']
        org_sent_labels_s = ex['org_sent_labels_s']
        poss_s = ex['poss_s']
        src_txt_s = ex['src_txt_s']
        tgt_txt = ex['tgt_txt']
        src_txt = ex['src_txt']

        if self.args.keep_orgdata:
            src = ex['src']
            src_sent_labels = ex['src_sent_labels']
            segs = ex['segs']
            if(not self.args.use_interval):
                segs = [0]*len(segs)
            clss = ex['clss']
            org_sent_labels = ex['org_sent_labels']
            poss = ex['poss']
            if is_test:
                return src_s,tgt,segs_s,clss_s,src_sent_labels_s,org_sent_labels_s,poss_s,src_txt_s,tgt_txt,src_txt, src,segs,clss,src_sent_labels,org_sent_labels,poss
                    # src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
            else:
                return src_s,tgt,segs_s,clss_s,src_sent_labels_s,org_sent_labels_s,poss_s,src,segs,clss,src_sent_labels,org_sent_labels,poss
        else:
            if is_test:
                return src_s,tgt,segs_s,clss_s,src_sent_labels_s,org_sent_labels_s,poss_s,src_txt_s,tgt_txt, src_txt
            else:
                return src_s,tgt,segs_s,clss_s,src_sent_labels_s,org_sent_labels_s,poss_s

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src_s'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            if self.args.task == 'abs':
                # (src_s, tgt, segs_s, clss_s, src_sent_labels_s, org_sent_labels_s, poss_s, src_txt_s, tgt_txt,
                #  src_txt), (src, segs, clss, src_sent_labels, org_sent_labels, poss)
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
            p_batch = self.batch(p_batch, self.batch_size)
            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test, self.args.keep_orgdata)

                yield batch
            return
