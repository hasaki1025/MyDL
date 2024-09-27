import logging
import os.path
from collections import defaultdict

import math
from nltk.translate.bleu_score import sentence_bleu

import torch
from torch import nn, no_grad

import EncoderDecoder
import Attention
from d2l.torch import d2l
from data import dataloader
from data.dataloader import Vocab




class Encoder(EncoderDecoder.Encoder):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.GRU = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x,*args):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x, state = self.GRU(x)
        return x, state


class Decoder(EncoderDecoder.Decoder):
    def __init__(self, hidden_size, vocab_size, num_layers, embedding_size, dropout=0.1):
        super(Decoder, self).__init__()
        self.context = None
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.GRU = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def init_context(self, enc_outputs):
        self.context = enc_outputs[1]

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, x: torch.Tensor, state: torch.Tensor,*args):
        x = self.embedding(x).permute(1, 0, 2)
        context = self.context[-1].repeat(x.shape[0], 1, 1)
        x_and_context = torch.concat([x, context], dim=-1)
        out, state = self.GRU(x_and_context, state)
        return self.linear(out).permute(1, 0, 2), state


class Seq2SeqEncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2SeqEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x):
        """
        :param enc_x: encoder inputs shape (batch_size, seq_len)
        :param dec_x: decoder inputs shape (batch_size, seq_len)
        :return: decoder outputs shape (batch_size, seq_len, vocab_size)
        """
        enc_out = self.encoder(enc_x)
        state = self.decoder.init_state(enc_out)
        self.decoder.init_context(enc_out)
        return self.decoder(dec_x, state)




def main():
    EncoderDecoder.logger.info("start")
    data_dir = './data/Multi30k/'
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
    patience = 10
    check_point_file = 'check_point/best_model.pth.tar'
    early_stopping = EncoderDecoder.EarlyStopping(patience=patience, save_file=check_point_file)
    train_iter, valid_iter, test_iter, src_vocab, tgt_vocab = dataloader.load_Multi30K_data(data_dir, num_steps,
                                                                                      batch_size)
    encoder = Encoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                      dropout)
    decoder = Decoder(num_hiddens, len(tgt_vocab), num_layers, embed_size, dropout)
    # attention = Attention.DotProductAttention(dropout)
    # decoder = Attention.Seq2SeqAttentionDecoder(num_hiddens, embed_size, len(tgt_vocab), num_layers, dropout,attention)
    net = Seq2SeqEncoderDecoder(encoder, decoder)
    if not os.path.exists(check_point_file):
        EncoderDecoder.train(net, train_iter, valid_iter, lr, num_epochs, tgt_vocab, early_stopping, device)
    else:
        state_dict = torch.load(check_point_file)
        net.load_state_dict(state_dict)
    EncoderDecoder.pred(net, num_steps, train_iter, tgt_vocab, device)


