

import math
import torch

from torch import nn

import EncoderDecoder
import Seq2Seq


class DotProductAttention(nn.Module):

    def __init__(self,dropout,**kwargs):
        super(DotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def input_mask(inputs,valid_lens):
        """
        :param inputs:  shape (batch_size, query_size, key_size)
        :param valid_lens: shape (batch_size,1)
        :return: shape (batch_size, query_size, key_size)
        """
        mask = (torch.arange(inputs.size(2),device=valid_lens.device).unsqueeze(0) < valid_lens.unsqueeze(-1)).unsqueeze(1).repeat(1, inputs.size(1),1)
        inputs[~mask] = -1e6
        return inputs



    def forward(self, q,k,v ,valid_lens=None):
        """
        :param q: shape (batch_size, query_size, d)
        :param k: shape (batch_size, key_size, d)
        :param v: shape (batch_size, key_size, value_len)
        :param valid_lens: shape (batch_size, 1)
        :return: shape(batch_size, query_size, value_len)
        """
        weight = torch.matmul(q,k.permute(0,2,1)) / math.sqrt(q.shape[-1])
        if valid_lens is not None:
            weight = self.input_mask(weight,valid_lens)
        score = self.softmax(weight)
        return torch.matmul(score,v)

class MultiHeadAttention(nn.Module):
    def __init__(self, query_len, key_len, value_len,nums_hidden,nums_head,attention,dropout=0.1,bias=False,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = attention
        self.nums_head = nums_head
        self.attention = DotProductAttention(dropout,**kwargs)
        self.W_q = nn.Linear(query_len, nums_hidden,bias)
        self.W_k = nn.Linear(key_len, nums_hidden,bias)
        self.W_v = nn.Linear(value_len, nums_hidden,bias)
        self.W_o = nn.Linear(nums_hidden, nums_hidden,bias)

    def transpose_qkv(self,x):
        """
        :param x: shape (batch_size, query_size or key_size, nums_hidden)
        :return: shape (batch_size * nums_head, query_size or key_size, nums_hidden)
        """
        x = x.reshape(x.shape[0],x.shape[1],self.nums_head,-1)
        x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], -1)
        return x


    def transpose_outputs(self,outputs):
        """
        :param outputs: (batch_size * nums_head, query_size, nums_hidden // nums_head)
        :return: shape (batch_size, query_size, nums_hidden)
        """
        outputs = outputs.reshape(-1,self.nums_head,outputs.shape[-2],outputs.shape[-1]).permute(0,2,1,3)
        return outputs.reshape(outputs.shape[0],outputs.shape[1],-1)



    def forward(self, q, k, v, valid_lens=None):
        """
        :param q: (batch_size, query_size, query_len)
        :param k: (batch_size, key_size, key_len)
        :param v: (batch_size, key_size, value_len)
        :param valid_lens: (batch_size, 1)
        :return: (batch_size, query_size, nums_hidden)
        """
        q,k,v = self.transpose_qkv(self.W_q(q)),self.transpose_qkv(self.W_k(k)),self.transpose_qkv(self.W_v(v))
        outputs = self.transpose_outputs(self.attention(q,k,v,valid_lens.repeat_interleave(self.nums_head)))
        return self.W_o(outputs)











