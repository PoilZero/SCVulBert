# coding:utf8
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from E50_utils.E55_Dataset import *
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

'''
    :return tokenlized dataset, tensorlized label
'''
def SCVul_collate_fn(batch_samples):
    batch_sentence = []
    batch_label = []
    for sample in batch_samples:
        batch_sentence.append(sample['sentence'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

if __name__ == "__main__":
    train_data = SCVulData()
    train_data_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)

    batch_X, batch_y = next(iter(train_data_loader))
    print(batch_X)
    print(batch_y)
    print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    print('batch_y shape:', batch_y.shape)

    # print(train_data_loader.dataset==train_data)