# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F

checkpoint = "bert-base-cased"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# hyper parameter
class Config():
    def __init__(self):
        self.dataset   = 'dataset/'+'reentrancy_273_directlly_from_dataset.txt'
        self.learning_rate = 1e-5
        self.epoch_num     = 250
        self.batch_size    = 16
        self.dropout       = 0.1
conf = Config()
