import os.path

import torch
from torch import nn

import EncoderDecoder
import Seq2Seq
from d2l.torch import d2l, bleu



def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测

    Defined in :numref:`sec_seq2seq_training`"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return tgt_vocab.list2sent(output_seq), attention_weight_seq


if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    train_data,valid_data,test_data,src_vocab,tar_vacab = Seq2Seq.dataloader.load_Multi30K_data('data/Multi30k',num_steps,batch_size)
    if not os.path.exists('d2lModel.pth'):
        encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                                dropout)
        decoder = d2l.Seq2SeqDecoder(len(tar_vacab), embed_size, num_hiddens, num_layers,
                                dropout)
        net = d2l.EncoderDecoder(encoder, decoder)
        d2l.train_seq2seq(net, train_data, lr, num_epochs, tar_vacab, device)
        torch.save(net, 'd2lModel.pth')
    else:
        net = torch.load('d2lModel.pth')
    src,tar = Seq2Seq.dataloader.load_data_from_file('test','data/Multi30k')
    scores = []

    for src, tar in zip(src, tar):
        translation, attention_weight_seq = predict_seq2seq(
            net, src, src_vocab, tar_vacab, num_steps, device)
        score = EncoderDecoder.cal_bleu(translation, tar)
        print(f'bleu {score:.3f}')
        scores.append(score)

    print(torch.tensor(scores).mean().item())



