# coding:utf8
import sys
import argparse
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
        self.batch_size    = 4
        self.dropout       = 0.
conf = Config()

def parse_arg():
    # conf.dataset = sys.argv[1] if len(sys.argv)>=2 else None

    parser = argparse.ArgumentParser(description='SCVulBert for Smart Contracts Vul Detection')
    parser.add_argument(
        'dataset', nargs='?'
        , default=conf.dataset
    )
    parser.add_argument(
        '--learning_rate', '-lr'
        , type=float, default=conf.learning_rate
    )
    parser.add_argument(
        '--epoch_num', '-e'
        , type=int, default=conf.epoch_num
    )
    parser.add_argument(
        '--batch_size', '-bs'
        , type=int, default=conf.batch_size
    )
    parser.add_argument(
        '--dropout', '-do'
        , type=float, default=conf.dropout
    )
    args = parser.parse_args()
    print('Using HyperParameter as follows:')
    for arg in vars(args):
        arg_value = getattr(args, arg)
        setattr(conf, arg, arg_value)
        print('-', arg, '=', arg_value)
parse_arg()