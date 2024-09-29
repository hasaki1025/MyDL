import itertools
import os
from collections import Counter
from enum import unique

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from triton.language import dtype


class Vocab:

    def __init__(self, data, min_freq=10):
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.begin_token = '<bos>'
        self.end_token = '<eos>'
        self.index2Token = None
        self.token2Index = None
        self.tokenFrequency = None
        vocab = dict()
        for sent in data:
            for word in sent:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        self.tokenFrequency = {key: value for key, value in vocab.items() if value >= min_freq}
        self.tokenFrequency['<unk>'] = min_freq
        self.tokenFrequency['<bos>'] = min_freq
        self.tokenFrequency['<pad>'] = min_freq
        self.tokenFrequency['<eos>'] = min_freq
        self.index2Token = list(self.tokenFrequency.keys())
        self.token2Index = {token: i for i, token in enumerate(self.index2Token)}

        self.begin_index = self.token2Index[self.begin_token]
        self.end_index = self.token2Index[self.end_token]
        self.unk_index = self.token2Index[self.unk_token]
        self.pad_index = self.token2Index[self.pad_token]

    def __len__(self):
        return len(self.index2Token)

    def token2index(self, token):
        return self.token2Index[token] if token in self.token2Index else self.token2Index[self.unk_token]

    def index2token(self, index):
        return self.index2Token[index]

    def get_token_freq(self, token):
        return self.tokenFrequency[token] if token in self.tokenFrequency else 0

    def sent2Vector(self, sent):
        return [self.token2index(token) for token in sent]


    def tensor2sent(self,tensor:torch.Tensor,valid_lens=None):
        """
        :param tensor: Tensor (batch_size,nums_step)
        :param valid_lens: shape (batch_size,1)
        :return: list[str] len = batch_size
        """
        sentences = []
        if valid_lens is not None:
            end_vector = torch.tensor([self.end_index]*tensor.shape[0], device=tensor.device).unsqueeze(-1)
            temp = torch.cat([tensor,end_vector],dim=-1)
            valid_lens = (temp == self.end_index).to(dtype=torch.int32).argmax(-1).tolist()
        else:
            valid_lens = [len(vector) for vector in tensor]
        for i,vector in enumerate(tensor):
            # sentences.append(self.vector2sent(vector[:valid_lens[i]]))
            sentences.append(self.vector2wordList(vector[:valid_lens[i]]))
        return sentences

    def vector2sent(self, vector):
        return self.list2sent(vector.tolist())

    def list2sent(self,l):
        return ' '.join([self.index2token(index) for index in l])

    def vector2wordList(self,vector):
        return [self.index2token(index) for index in vector.tolist()]

    def __getitem__(self, item):
        if isinstance(item, list):
            return self.sent2Vector(item)
        elif isinstance(item, int):
            return self.index2token(item)
        elif isinstance(item, str):
            return self.token2index(item)
        else:
            raise TypeError


def read_data_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()


def read_data(lang, data_dir):
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.' + lang):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                data += f.readlines()
    return data


def tokenize_sentence(sentences):
    return [sent.split() for sent in sentences]


def preprocess(src_data, tar_data):
    src_data = tokenize_sentence(src_data)
    tar_data = tokenize_sentence(tar_data)
    return src_data, tar_data


class TranslateDataset(Dataset):

    def __init__(self, src_data, tar_data):
        super(TranslateDataset, self).__init__()
        self.src_data = src_data
        self.tar_data = tar_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, item):
        return self.src_data[item], self.tar_data[item]


# class ProcessBatch:
#     def __init__(self, nums_step, src_vocab, tar_vocab):
#         self.nums_step = nums_step
#         self.src_vocab = src_vocab
#         self.tar_vocab = tar_vocab
#
#     def truncate_and_pad_sentence(self, sentence, vocab: Vocab):
#         sentence = sentence[:self.nums_step - 1] + [vocab.end_token]
#         padding_len = self.nums_step - len(sentence)
#         return sentence + [vocab.pad_token] * padding_len, self.nums_step - padding_len
#
#     def __call__(self, batch):
#         batch = [
#             (self.truncate_and_pad_sentence(src, self.src_vocab), self.truncate_and_pad_sentence(tar, self.tar_vocab))
#             for src, tar in batch]  # [(src_sent, valid_len),()]
#         src_data = [item[0][0] for item in batch]
#         src_val_len = [item[0][1] for item in batch]
#         tar_data = [item[1][0] for item in batch]
#         tar_val_len = [item[1][1] for item in batch]
#         src_data, tar_data = [self.src_vocab[sent] for sent in src_data], [self.tar_vocab[sent] for sent in tar_data]
#         return (torch.tensor(src_data),
#                 torch.tensor(src_val_len),
#                 torch.tensor(tar_data),
#                 torch.tensor(tar_val_len))

class ProcessBatch:
    def __init__(self, nums_step, vocab):
        self.nums_step = nums_step
        self.vocab = vocab

    def truncate_and_pad_sentence(self, sentence, vocab: Vocab):
        sentence = sentence[:self.nums_step - 1] + [vocab.end_token]
        padding_len = self.nums_step - len(sentence)
        return sentence + [vocab.pad_token] * padding_len, self.nums_step - padding_len

    def __call__(self, batch):
        batch = [
            (self.truncate_and_pad_sentence(src, self.vocab), self.truncate_and_pad_sentence(tar, self.vocab))
            for src, tar in batch]  # [(src_sent, valid_len),()]
        src_data = [item[0][0] for item in batch]
        src_val_len = [item[0][1] for item in batch]
        tar_data = [item[1][0] for item in batch]
        tar_val_len = [item[1][1] for item in batch]
        src_data, tar_data = [self.vocab[sent] for sent in src_data], [self.vocab[sent] for sent in tar_data]
        return (torch.tensor(src_data),
                torch.tensor(src_val_len),
                torch.tensor(tar_data),
                torch.tensor(tar_val_len))



def split_data(data, train_ratio=0.8, val_ratio=0.1):
    nums_train = int(train_ratio * len(data))
    nums_val = int(val_ratio * len(data))
    return data[:nums_train], data[nums_train: nums_train + nums_val], data[nums_train + nums_val:]


def check_data_size(dataset, nums_step):
    for i in range(len(dataset)):
        src_data, src_valid_len, tar_data, tar_valid_len = dataset[i]
        if len(src_data) != nums_step and len(tar_data) != nums_step:
            print(i, len(src_data), len(tar_data))
            raise ValueError

def load_data_from_file(data_type,data_dir):
    with open('{}/{}.{}'.format(data_dir,data_type,'en'), 'r', encoding='utf-8') as f:
        src_data = f.readlines()
    with open('{}/{}.{}'.format(data_dir,data_type,'de'), 'r', encoding='utf-8') as f:
        tar_data = f.readlines()
    return src_data, tar_data

# def load_Multi30K_data(data_dir, nums_step, batch_size):
#     # src_data, tar_data = preprocess(read_data('de', data_dir), read_data('en', data_dir))
#     train_src, train_tar = preprocess(*load_data_from_file('train', data_dir))
#     val_src, val_tar = preprocess(*load_data_from_file('val', data_dir))
#     test_src, test_tar = preprocess(*load_data_from_file('test', data_dir))
#
#
#     src_vocab, tar_vocab = Vocab(train_src + val_src + test_src), Vocab(train_tar + val_tar + test_tar)
#     train_data_set, val_data_set, test_data_set = (
#         TranslateDataset(train_src, train_tar),
#         TranslateDataset(val_src, val_tar),
#         TranslateDataset(test_src, test_tar))
#
#
#     processor = ProcessBatch(nums_step, src_vocab, tar_vocab)
#     return (DataLoader(train_data_set, batch_size=batch_size, shuffle=True, collate_fn=processor),
#             DataLoader(val_data_set, batch_size=batch_size, shuffle=True, collate_fn=processor),
#             DataLoader(test_data_set, batch_size=batch_size, shuffle=True, collate_fn=processor), src_vocab, tar_vocab)



def load_Multi30K_data(data_dir, nums_step, batch_size):
    # src_data, tar_data = preprocess(read_data('de', data_dir), read_data('en', data_dir))
    train_src, train_tar = preprocess(*load_data_from_file('train', data_dir))
    val_src, val_tar = preprocess(*load_data_from_file('val', data_dir))
    test_src, test_tar = preprocess(*load_data_from_file('test', data_dir))


    vocab = Vocab(train_src + val_src + test_src + train_tar + val_tar + test_tar)
    train_data_set, val_data_set, test_data_set = (
        TranslateDataset(train_src, train_tar),
        TranslateDataset(val_src, val_tar),
        TranslateDataset(test_src, test_tar))


    processor = ProcessBatch(nums_step, vocab)
    return (DataLoader(train_data_set, batch_size=batch_size, shuffle=True, collate_fn=processor),
            DataLoader(val_data_set, batch_size=batch_size, shuffle=True, collate_fn=processor),
            DataLoader(test_data_set, batch_size=batch_size, shuffle=True, collate_fn=processor), vocab)


def get_seq_arg_len():
    with open('Multi30k/train.de','r', encoding='utf-8') as f:
        lines = f.readlines()
        tokens = 0
        max_len = 0
        min_len = 100000
        for line in lines:
            token_list = tokenize_sentence(line)
            tokens += len(token_list)
            max_len = max(max_len, len(token_list))
            min_len = min(min_len, len(token_list))

        return tokens/ len(lines), max_len, min_len
# load_Multi30K_data('Multi30k', 10, 64)

# print(get_seq_arg_len())