import os

import math
import torch


from torch import nn


import EncoderDecoder
from d2l.torch import reshape, d2l
from data import dataloader
from data.dataloader import Vocab


class DotProductAttention(nn.Module):

    def __init__(self,**kwargs):
        super(DotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, q,k,v ,mask=None):
        """
        :param q: shape (batch_size, query_size, d)
        :param k: shape (batch_size, key_size, d)
        :param v: shape (batch_size, key_size, value_len)
        :param mask: shape (batch_size, query_size, key_size)
        :return: shape(batch_size, query_size, value_len)
        """
        weight = torch.matmul(q,k.permute(0,2,1)) / math.sqrt(q.shape[-1])
        if mask is not None:
            weight = weight.masked_fill(mask == 0, float('-inf'))

        score = self.softmax(weight)
        return torch.matmul(score,v)



class MultiHeadAttention(nn.Module):
    def __init__(self, query_len, key_len, value_len,nums_hidden,nums_head,dropout=0.1,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.nums_head = nums_head
        self.attention = DotProductAttention(**kwargs)
        self.W_q = nn.Linear(query_len, nums_hidden)
        self.W_k = nn.Linear(key_len, nums_hidden)
        self.W_v = nn.Linear(value_len, nums_hidden)
        self.W_o = nn.Linear(nums_hidden, nums_hidden)
        self.dropout = nn.Dropout(dropout)

    def transpose_qkv(self,x):
        """
        :param x: shape (batch_size, query_size or key_size, nums_hidden)
        :return: shape (batch_size * nums_head, query_size or key_size, nums_hidden)
        """
        x = x.reshape(x.shape[0],x.shape[1],self.nums_head,-1)
        x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], -1)
        return self.dropout(x)


    def transpose_outputs(self,outputs):
        """
        :param outputs: (batch_size * nums_head, query_size, nums_hidden // nums_head)
        :return: shape (batch_size, query_size, nums_hidden)
        """
        outputs = outputs.reshape(-1,self.nums_head,outputs.shape[-2],outputs.shape[-1]).permute(0,2,1,3)
        return outputs.reshape(outputs.shape[0],outputs.shape[1],-1)



    def forward(self, q, k, v, mask=None):
        """
        :param q: (batch_size, query_size, query_len)
        :param k: (batch_size, key_size, key_len)
        :param v: (batch_size, key_size, value_len)
        :param mask: (batch_size, query_size, key_size)
        :return: (batch_size, query_size, nums_hidden)
        """
        q,k,v = self.transpose_qkv(self.W_q(q)),self.transpose_qkv(self.W_k(k)),self.transpose_qkv(self.W_v(v))
        if mask is not None:
            """这里只需要重复第一维因为transpose_qkv只修改了qkv最后一维的长度，最后一维的长度并不会影响最后注意力得分矩阵的形状"""
            mask = torch.repeat_interleave(mask,self.nums_head,dim=0)
        outputs = self.transpose_outputs(self.attention(q,k,v,mask))
        return self.W_o(outputs)




class AddNorm(nn.Module):

    def __init__(self,normalize_shape,dropout=0.1,**kwargs):
        super(AddNorm,self).__init__(**kwargs)
        self.norm = nn.LayerNorm(normalized_shape=normalize_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x ,y):
        return self.norm(x+self.dropout(y))


class PositionWiseFFN(nn.Module):

    def __init__(self,input_size,hidden_size,output_size,**kwargs):
        super(PositionWiseFFN,self).__init__(**kwargs)
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        return self.linear2(self.relu(self.linear1(x)))

# TODO 为什么对整个样本规范化而不是对单独一行进行规范化

class TransformerEncoderBlock(nn.Module):

    def __init__(self,d_model,norm_shape,ffn_hidden,nums_head,dropout=0.1,**kwargs):
        super(TransformerEncoderBlock,self).__init__(**kwargs)
        self.multiAttention = MultiHeadAttention(d_model,d_model,d_model,d_model,nums_head,dropout,**kwargs)
        self.norm1 = AddNorm(normalize_shape=norm_shape,dropout=dropout,**kwargs)
        self.ffn = PositionWiseFFN(d_model,ffn_hidden,d_model,**kwargs)
        self.norm2 = AddNorm(normalize_shape=norm_shape,dropout=dropout,**kwargs)

    def forward(self,x,mask=None):
        """
        :param x: shape (batch_size, max_len, d_model)
        :param mask: shape (batch_size, max_len, max_len)
        :return: shape (batch_size, max_len, d_model)
        """

        attention_out = self.multiAttention(x,x,x,mask)
        x = self.norm1(x,attention_out)
        ffn_out = self.ffn(x)
        return self.norm2(x,ffn_out)


class PositionEmbedding(nn.Module):
    def __init__(self,dropout,**kwargs):
        super(PositionEmbedding,self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout) # TODO 为什么要添加Dropout
        self.position_cache = dict()

    def forward(self,x):
        """
        :param x: shape (batch_size, max_len, d_model)
        :return: shape (batch_size, max_len, d_model)
        """
        shape = (x.shape[1],x.shape[2])
        if shape not in self.position_cache.keys():
            d = x.shape[-1]
            max_len = x.shape[-2]
            row = torch.arange(max_len, dtype=torch.float32).reshape(max_len, 1).repeat(1, d)
            col = torch.arange(d, dtype=torch.float32).repeat(max_len, 1)
            col[:, 1::2] -= 1
            position = row / (10000 ** (2 * col / d))
            position[:, 0::2] = torch.sin(position[:, 0::2])
            position[:, 1::2] = torch.cos(position[:, 1::2])
            self.position_cache[shape] = position.unsqueeze(0).to(x.device)
        return self.dropout(x + self.position_cache[shape])



def valid_lens2mask(query_size,key_size,valid_lens,mask_after=False):
    """
    :param query_size: query.shape[1]
    :param key_size: key.shape[1]
    :param valid_lens: shape (batch_size,)
    :param mask_after: mask matrix tril
    :return: shape (batch_size, query_size, key_size)
    """
    if valid_lens is None :
        return torch.ones([1,query_size,key_size])

    batch_size = valid_lens.shape[0]
    mask = (torch.arange(key_size,device=valid_lens.device) < valid_lens.unsqueeze(-1)).reshape(batch_size,1,-1).repeat(1,query_size,1)
    return mask if not mask_after else mask & torch.ones_like(mask).tril()


class TransformerEncoder(nn.Module):
    def __init__(self,embedding,d_model,norm_shape,ffn_hidden,nums_head,nums_layer,dropout=0.1,**kwargs):
        super(TransformerEncoder,self).__init__(**kwargs)
        self.embedding = embedding
        self.position_embedding = PositionEmbedding(dropout,**kwargs)
        self.TransformerEncoderBlocks = nn.ModuleList([TransformerEncoderBlock(d_model,norm_shape,ffn_hidden,nums_head,dropout=dropout,**kwargs) for _ in range(nums_layer)])




    def forward(self,x,mask):
        """
        :param x: shape (batch_size, max_len, vocab_size)
        :param valid_lens: (batch_size,)
        :return: shape (batch_size, max_len, d_model)
        """
        x = self.position_embedding(self.embedding(x))

        for block in self.TransformerEncoderBlocks:
            x = block(x,mask)
        return x



class TransformerDecoderBlock(nn.Module):

    def __init__(self,d_model,norm_shape,ffn_hidden,nums_head,dropout=0.1,**kwargs):
        super(TransformerDecoderBlock,self).__init__(**kwargs)
        self.multiHeadAttention1 = MultiHeadAttention(d_model,d_model,d_model,d_model,nums_head,dropout=dropout)
        self.norm1 = AddNorm(normalize_shape=norm_shape,dropout=dropout)
        self.multiHeadAttention2 = MultiHeadAttention(d_model,d_model,d_model,d_model,nums_head,dropout=dropout)
        self.norm2 = AddNorm(normalize_shape=norm_shape,dropout=dropout)
        self.ffn = PositionWiseFFN(d_model,ffn_hidden,d_model,**kwargs)
        self.norm3 = AddNorm(normalize_shape=norm_shape,dropout=dropout)

    def forward(self,x,enc_outputs,enc_mask,dec_mask=None):
        """
        :param x: shape (batch_size, max_len, d_model)
        :param enc_outputs: (batch_size, max_len, d_model)
        :param enc_mask: (batch_size,max_len,max_len) or None
        :param dec_mask: (batch_size,max_len,max_len)
        :return: shape (batch_size, max_len, d_model)
        """
        y1 = self.multiHeadAttention1(x,x,x,dec_mask) # TODO 这里的Mask和下面的Mask不同 ?
        x = self.norm1(x,y1)
        y2 = self.multiHeadAttention2(q=x,k=enc_outputs,v=enc_outputs,mask=enc_mask)
        x = self.norm2(x=x,y=y2)
        y3 = self.ffn(x)
        return self.norm3(x=x,y=y3)




class TransformerDecoder(nn.Module):

    def __init__(self,embedding,vocab_size,d_model,norm_shape,ffn_hidden,nums_head,nums_layer,dropout=0.1,**kwargs):
        super(TransformerDecoder,self).__init__()
        self.embedding = embedding
        self.position_embedding = PositionEmbedding(dropout,**kwargs)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(d_model,norm_shape,ffn_hidden,nums_head,dropout=dropout,**kwargs) for _ in range(nums_layer)])
        self.linear = nn.Linear(d_model,vocab_size)


    def forward(self,x,enc_outputs,enc_mask=None,dec_mask=None):
        """
        :param x: shape (batch_size, max_len, vocab_size)
        :param enc_outputs: shape (batch_size, max_len, d_model)
        :param valid_lens: shape (batch_size,)
        :return: shape (batch_size, max_len, vocab_size)
        """
        x = self.position_embedding(self.embedding(x))
        for layer in self.decoder_layers:
            x = layer(x,enc_outputs,enc_mask,dec_mask)
        return self.linear(x)


# class TransformerModel(nn.Module):
#     def __init__(self,d_model,norm_shape,ffn_hidden,nums_head,nums_layer,src_vocab:Vocab,tgt_vocab:Vocab,dropout=0.1,**kwargs):
#         super(TransformerModel,self).__init__(**kwargs)
#         self.tgt_vocab = tgt_vocab
#         self.encoder:TransformerEncoder = TransformerEncoder(
#             len(src_vocab),
#             d_model=d_model,
#             norm_shape=norm_shape,
#             ffn_hidden=ffn_hidden,
#             nums_head=nums_head,
#             nums_layer=nums_layer,
#             dropout=dropout,
#             **kwargs
#         )
#         self.decoder:TransformerDecoder = TransformerDecoder(
#             len(tgt_vocab),
#             d_model=d_model,
#             norm_shape=norm_shape,
#             ffn_hidden=ffn_hidden,
#             nums_head=nums_head,
#             nums_layer=nums_layer,
#             dropout=dropout,
#             **kwargs
#         )
#
#     def forward(self,batch):
#         """
#         :param batch: x shape == (batch_size, max_len) ,y shape == (batch_size, max_len) ,valid_lens shape == (batch_size,)
#         :return: train shape (batch_size, max_len, vocab_size) and pred shape (batch_size, max_len)
#         """
#         x,x_valid_lens,y,y_valid_lens = batch
#         enc_mask0 = valid_lens2mask(x.shape[1],x.shape[1],x_valid_lens)
#         enc_output = self.encoder(x,enc_mask0)
#         if self.training:
#             dec_input = EncoderDecoder.generate_train_decoder_inputs(y, self.tgt_vocab.begin_index)
#             dec_mask = valid_lens2mask(y.shape[1],y.shape[1],y_valid_lens + 1,mask_after=True)
#             """这里可以直接使用Encoder之前使用的mask是因为enc_output.shape[1]==y.shape[1]"""
#             return self.decoder(dec_input,enc_output,enc_mask0,dec_mask)
#         else:
#             max_len = y.size(1)
#             batch_size = y.size(0)
#             """这里需要重新生成Encoder的Mask因为query_size！= key_size,此时的query_size为1，key_size依旧为max_len"""
#             enc_mask = valid_lens2mask(1, enc_output.shape[1], x_valid_lens, mask_after=False)
#             dec_input = torch.full((batch_size,1), self.tgt_vocab.begin_index,device=y.device)
#             for i in range(max_len):
#                 dec_out = EncoderDecoder.pred_softmax(
#                     self.decoder(dec_input,enc_output,enc_mask,None)
#                 )
#                 next_token = dec_out[:,-1].unsqueeze(-1)
#                 dec_input = torch.concat([dec_input,next_token],dim=-1)
#             return dec_input[:,1:]

class TransformerModel(nn.Module):
    def __init__(self,d_model,norm_shape,ffn_hidden,nums_head,nums_layer,vocab:Vocab,dropout=0.1,**kwargs):
        super(TransformerModel,self).__init__(**kwargs)
        self.vocab = vocab
        embedding = nn.Embedding(len(vocab),d_model)
        self.encoder:TransformerEncoder = TransformerEncoder(
            embedding,
            d_model=d_model,
            norm_shape=norm_shape,
            ffn_hidden=ffn_hidden,
            nums_head=nums_head,
            nums_layer=nums_layer,
            dropout=dropout,
            **kwargs
        )
        self.decoder:TransformerDecoder = TransformerDecoder(
            embedding,
            len(vocab),
            d_model=d_model,
            norm_shape=norm_shape,
            ffn_hidden=ffn_hidden,
            nums_head=nums_head,
            nums_layer=nums_layer,
            dropout=dropout,
            **kwargs
        )

    def forward(self,batch):
        """
        :param batch: x shape == (batch_size, max_len) ,y shape == (batch_size, max_len) ,valid_lens shape == (batch_size,)
        :return: train shape (batch_size, max_len, vocab_size) and pred shape (batch_size, max_len)
        """
        x,x_valid_lens,y,y_valid_lens = batch
        enc_mask0 = valid_lens2mask(x.shape[1],x.shape[1],x_valid_lens)
        enc_output = self.encoder(x,enc_mask0)
        if self.training:
            dec_input = EncoderDecoder.generate_train_decoder_inputs(y, self.vocab.begin_index)
            dec_mask = valid_lens2mask(y.shape[1],y.shape[1],y_valid_lens + 1,mask_after=True)
            """这里可以直接使用Encoder之前使用的mask是因为enc_output.shape[1]==y.shape[1]"""
            return self.decoder(dec_input,enc_output,enc_mask0,dec_mask)
        else:
            max_len = y.size(1)
            batch_size = y.size(0)
            """这里需要重新生成Encoder的Mask因为query_size！= key_size,此时的query_size为1，key_size依旧为max_len"""
            enc_mask = valid_lens2mask(1, enc_output.shape[1], x_valid_lens, mask_after=False)
            dec_input = torch.full((batch_size,1), self.vocab.begin_index,device=y.device)
            for i in range(max_len):
                dec_out = EncoderDecoder.pred_softmax(
                    self.decoder(dec_input,enc_output,enc_mask,None)
                )
                next_token = dec_out[:,-1].unsqueeze(-1)
                dec_input = torch.concat([dec_input,next_token],dim=-1)
            return dec_input[:,1:]











# def main():
#     EncoderDecoder.logger.info("start")
#     data_dir = './data/Multi30k/'
#     num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 12
#     lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
#     ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
#     norm_shape = [32]
#     patience = 10
#     check_point_file = 'check_point/best_transformer_model.pth.tar'
#     early_stopping = EncoderDecoder.EarlyStopping(patience=patience,delta=0.001, save_file=check_point_file)
#     train_iter, valid_iter, test_iter, src_vocab, tgt_vocab = dataloader.load_Multi30K_data(data_dir, num_steps,batch_size)
#     net = TransformerModel(num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,src_vocab,tgt_vocab,dropout)
#
#     if not os.path.exists(check_point_file):
#         EncoderDecoder.train(net, train_iter, valid_iter, lr, num_epochs, tgt_vocab, early_stopping, device)
#     else:
#         state_dict = torch.load(check_point_file)
#         net.load_state_dict(state_dict)
#     EncoderDecoder.pred(net, test_iter, tgt_vocab, device)



def main():
    EncoderDecoder.logger.info("start")
    data_dir = './data/Multi30k/'
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 12
    lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    norm_shape = [32]
    patience = 10
    check_point_file = 'check_point/best_transformer_model2.pth.tar'
    early_stopping = EncoderDecoder.EarlyStopping(patience=patience,delta=0.001, save_file=check_point_file)
    train_iter, valid_iter, test_iter, vocab = dataloader.load_Multi30K_data(data_dir, num_steps,batch_size)
    net = TransformerModel(num_hiddens,norm_shape,ffn_num_hiddens,num_heads,num_layers,vocab,dropout)

    if not os.path.exists(check_point_file):
        EncoderDecoder.train(net, train_iter, valid_iter, lr, num_epochs, vocab, early_stopping, device)
    else:
        state_dict = torch.load(check_point_file)
        net.load_state_dict(state_dict)
    EncoderDecoder.pred(net, test_iter, vocab, device)
