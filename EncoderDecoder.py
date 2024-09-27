import collections
import logging

import math
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import torch
from torch import nn, no_grad


from d2l.torch import d2l
from data.dataloader import Vocab

# 创建一个logger
logger = logging.getLogger('MyLogger')
logger.setLevel(logging.INFO)  # 设置日志级别

# 创建一个handler，用于写入日志文件
file_handler = logging.FileHandler('app.log', mode='a')
file_handler.setLevel(logging.INFO)

# 创建一个handler，用于输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(file_handler)
logger.addHandler(console_handler)




class Encoder(nn.Module):

    def __init__(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)



    def forward(self,x,*args):
        """
        :param x: shape (batch_size, seq_len)
        :param args: else args
        :return: outputs shape (nums_step,batch_size,nums_hidden) and states shape (nums_layer, batch_size, nums_hidden)
        """
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)

    def init_state(self,enc_outputs):
        """
        :param enc_outputs: enc_outputs and states
        :return: default is states
        """
        raise NotImplementedError

    def forward(self,x,enc_state,*args):
        """
        :param x: train data shape is (batch_size, seq_len)
        :param enc_state: shape (nums_layer, batch_size, nums_hidden)
        :param args: else args
        :return: outputs shape (batch_size, nums_step, vocab_size)
        """
        raise NotImplementedError



def generate_mask(shape, valid_len):
    """
    :param shape: loss shape (batch_size, seq_len)
    :param valid_len: shape (batch_size,)
    :return: mask ,True keep and False throw
    """
    mask = torch.arange(shape[1], device=valid_len.device) < valid_len.reshape(-1, 1)
    return mask.to(dtype=torch.float32)

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):



    def forward(self, inputs, targets, valid_lens):
        """
        :param inputs: shape (batch_size, seq_len, vocab_size)
        :param targets: shape (batch_size, seq_len)
        :param valid_lens: shape (batch_size,)
        :return: loss shape (batch_size, seq_len)
        """
        self.reduction = 'none'
        unweight_loss = super(MaskedSoftmaxCELoss, self).forward(inputs.permute(0, 2, 1), targets)
        mask = generate_mask(unweight_loss.shape, valid_lens)
        return unweight_loss * mask



class EarlyStopping:
    def __init__(self, patience=10,delta=0.1,save_file='check_point/best_model.pth.tar'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_file = save_file

    def __call__(self, val_loss,model):

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model,val_loss)

        if val_loss + self.delta > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(model,val_loss)

        return self.early_stop

    def save_checkpoint(self, model,loss):
        logger.info("Get better model ,loss {}, save model to {}".format(loss,self.save_file))
        torch.save(model.state_dict(), self.save_file)



def calculate_bleu(pred,target,valid_len,vocab:Vocab):
    """
    :param pred: shape (batch_size, seq_len)
    :param target: shape (batch_size, seq_len)
    :param valid_len: shape (batch_size,)
    :param vocab: target vocab
    :return: float score
    """
    batch_size = pred.size(0)
    pred_sentences = vocab.tensor2sent(pred)
    target_sentences = vocab.tensor2sent(target,valid_lens=valid_len)
    return sum([ sentence_bleu([target_sentences[i]], pred_sentences[i],weights=[1,0,0,0]) for i in range(batch_size)]) / batch_size

def cal_bleu(pred,target):
    """
    :param pred: str
    :param target: str
    :return: float score
    """
    pred ,target = pred.split(' '), target.split(' ')
    return sentence_bleu([target], pred, weights=[1,0,0,0])



def generate_train_decoder_inputs(target,begin_index):
    """
    :param target: shape (batch_size, seq_len)
    :param begin_index: int from vocab mean seq start ,like <bos>
    :return: [bos_vector,target] else [bos_vector]
    """
    begin_vector = torch.tensor([begin_index]*target.shape[0],device=target.device).unsqueeze(-1)
    return torch.concat([begin_vector, target[:,:-1]], dim=-1) if target is not None else begin_vector.reshape(-1,1)

def calculate_acc(pred:torch.Tensor, target,valid_lens):
    """
    :param pred: shape (batch_size, seq_len, vocab_size)
    :param target: shape (batch_size, seq_len)
    :return: float
    """
    pred = nn.functional.softmax(pred, dim=-1)
    mask = generate_mask(target.shape, valid_lens)
    return ((pred.argmax(dim=-1).reshape(pred.shape[0], -1) == target) * mask ).sum().item() / (mask.sum().item())

def evaluate_epoch(valid_iter,net,criterion,begin_index,device):
    """
    :param valid_iter: valid data iter ,batch shape (batch_size, seq_len)
    :param net: model
    :param criterion: loss function
    :param begin_index: seq start id
    :return: arg epoch loss and arg epoch acc
    """
    epoch_loss = 0.0
    epoch_acc = 0
    nums_batch = 0
    for batch in valid_iter:
        with no_grad():
            x, x_valid_lens, y, y_valid_lens = [d.to(device) for d in batch]
            de_x = generate_train_decoder_inputs(y, begin_index)
            y_hat, _ = net(x, de_x)
            nums_batch += 1
            epoch_acc += calculate_acc(y_hat, y,y_valid_lens)
            epoch_loss += criterion(y_hat, y, y_valid_lens).sum().item()
    return epoch_loss / nums_batch, epoch_acc / nums_batch

def train(net, train_iter,valid_iter, lr, num_epoch, tar_vocab:Vocab, early_stop,device='cuda'):
    """
    :param net: input shape(batch_size, seq_len) ---> output shape(batch_size, seq_len, vocab_size)
    :param train_iter: train data iter ,batch shape (batch_size, seq_len)
    :param valid_iter: valid data iter ,batch shape (batch_size, seq_len)
    :param lr: learning rate
    :param num_epoch: epochs
    :param tar_vocab: vocab of target language
    :param early_stop: early stop class
    :param device: device
    :return: None
    """
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)
    criterion = MaskedSoftmaxCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    begin_index = tar_vocab.begin_index
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epoch])
    early_stopping = early_stop
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        epoch_acc = 0
        nums_batch = 0
        for batch in train_iter:
            optimizer.zero_grad()
            x, x_valid_lens, y, y_valid_lens = [d.to(device) for d in batch]
            de_x = generate_train_decoder_inputs(y, begin_index)
            y_hat, _ = net(x, de_x)
            loss = criterion(y_hat, y, y_valid_lens).sum()
            loss.backward()
            d2l.grad_clipping(net, 1)
            num_tokens = y_valid_lens.sum()
            optimizer.step()
            with torch.no_grad():
                nums_batch += 1
                metric.add(loss.item(), num_tokens)
                epoch_acc += calculate_acc(y_hat, y,y_valid_lens)
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
        logger.info('Epoch: {}, Train Loss: {}, Train ACC: {}'.format(epoch, epoch_loss/nums_batch, epoch_acc/nums_batch))
        valid_loss, valid_acc = evaluate_epoch(valid_iter,net,criterion,begin_index,device)
        logger.info('Epoch: {},Valid Loss: {},Valid ACC: {}'.format(epoch, valid_loss, valid_acc))
        if early_stopping(valid_loss,net):
            logger.info('Early Stopping in epoch {}...'.format(epoch))
            break

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


def pred_softmax(pred):
    """
    :param pred: input shape (batch_size,1,vocab_size)
    :return: shape (batch_size,1)
    """
    softmax = nn.Softmax(dim=-1)
    pred = softmax(pred).argmax(dim=-1)
    return pred.reshape(pred.shape[0], -1)


def pred(net, nums_step, data_iter, tar_vocab:Vocab,device='cuda'):
    """
    :param net: input shape(batch_size, seq_len) ---> output shape(batch_size, seq_len, vocab_size)
    :param nums_step: seq len
    :param data_iter: test data iter ,batch shape (1, seq_len) batch_size must be 1
    :param tar_vocab: target language vocab
    :param device: device
    :return: None
    """
    net.to(device)
    net.eval()
    logger.info("start test...")
    begin_index = tar_vocab.begin_index
    end_index = tar_vocab.end_index
    score_sum = 0.0
    num_batch = 0
    for batch in data_iter:
        with torch.no_grad():
            x, x_valid_lens, y, y_valid_lens = [d.to(device) for d in batch]
            enc_output = net.encoder(x)
            state = net.decoder.init_state(enc_output)
            net.decoder.init_context(enc_output)
            de_x = torch.tensor([begin_index]*y.shape[0],device=device).unsqueeze(-1)
            batch_pred = []
            for i in range(nums_step):
                y_hat, state = net.decoder(de_x,state)
                y_hat = pred_softmax(y_hat)
                if y_hat.shape[0] == 1 and y_hat.item() == end_index:
                    break
                batch_pred.append(y_hat)
                de_x = y_hat


        pred_tensor = torch.concat(batch_pred, dim=-1)
        score = calculate_bleu(pred_tensor, y, y_valid_lens, tar_vocab)
        score_sum += score
        num_batch += 1
        logger.info("batch bleu: {}".format(score))

    logger.info("test arg bleu score: {}".format(score_sum/num_batch))