# coding:utf8

import torch
from torch import nn

checkpoint = "bert-base-cased"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# super parameter
class Config():
    def __init__(self):
        self.dataset   = 'dataset/'+'reentrancy_273.txt'
        self.learning_rate = 1e-5
        self.epoch_num     = 20
        self.batch_size    = 16
conf = Config()
