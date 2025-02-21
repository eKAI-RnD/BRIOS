import torch
import torch.nn as nn


from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.utils as nn_utils
import math
from models import rios_lst
from sklearn import metrics

# default parameters
# SEQ_LEN = 46
# RNN_HID_SIZE = 64


class Model(nn.Module):
    def __init__(self, rnn_hid_size, INPUT_SIZE, SEQ_LEN, SELECT_SIZE):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.SEQ_LEN = SEQ_LEN
        self.SELECT_SIZE = SELECT_SIZE
        self.INPUT_SIZE = INPUT_SIZE

        self.build()

    def build(self):
        self.rios_f = rios_lst.Model(self.rnn_hid_size, self.INPUT_SIZE, self.SEQ_LEN, self.SELECT_SIZE)
        self.rios_b = rios_lst.Model(self.rnn_hid_size, self.INPUT_SIZE, self.SEQ_LEN, self.SELECT_SIZE)

    def forward(self, data):
        ret_f = self.rios_f(data, 'forward')
        ret_b = self.reverse(self.rios_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b)
        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss

        ret_f['imputations_f'] = ret_f['imputations']
        ret_f['imputations_b'] = ret_b['imputations']

        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if isinstance(tensor_, torch.Tensor):  # Kiểm tra nếu tensor_ là một tensor
                if tensor_.dim() <= 1:
                    return tensor_
                indices = range(tensor_.size()[1])[::-1]
                indices = torch.LongTensor(indices)
                if torch.cuda.is_available():
                    indices = indices.cuda()
                return tensor_.index_select(1, indices)
            else:
                # Nếu tensor_ là một số thực (float), trả về chính nó mà không làm gì thêm
                return tensor_

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer, epoch=None, max_norm=1.0):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            nn_utils.clip_grad_norm_(self.parameters(), max_norm)
            optimizer.step()

        return ret

