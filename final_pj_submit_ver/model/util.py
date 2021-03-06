# -*- coding:utf8 -*-
from torchtext.data import Iterator, BucketIterator
from torchtext import data
import torch

def load_iters(batch_size=32, device="cpu", data_path='data', vectors=None, use_tree=False):
    if not use_tree:
        TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
        LABEL = data.LabelField(batch_first=True)
        TREE = None

        fields = {'sentence1': ('premise', TEXT),
                  'sentence2': ('hypothesis', TEXT),
                  'label': ('label', LABEL)}

    train_data, dev_data, test_data = data.TabularDataset.splits(
        path=data_path,
        train='train.json',
        validation='dev.json',
        test='test.json',
        format='json',
        fields=fields,
        filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
    )
    if vectors is not None:
        TEXT.build_vocab(train_data, vectors=vectors, unk_init=torch.Tensor.normal_)
    else:
        TEXT.build_vocab(train_data)
    LABEL.build_vocab(dev_data)
    

    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.premise) + len(x.hypothesis),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(test_data,
                         batch_size=batch_size,
                         device=device,
                         sort=False,
                         sort_within_batch=False,
                         repeat=False,
                         shuffle=False)

    return train_iter, dev_iter, test_iter, TEXT, LABEL, TREE