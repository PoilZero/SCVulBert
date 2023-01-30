# coding:utf8

from E50_com import *
from transformers import AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import BertForSequenceClassification


class AttBLSTM(nn.Module):
    def __init__(self, input_size, label_size, batch_size=conf.batch_size, num_layer=1):
        super(AttBLSTM, self).__init__()
        self.num_layer  = num_layer
        self.batch_size = batch_size
        self.input_size = input_size
        self.h0_var, self.c0_var = None, None

        self.blstm = torch.nn.LSTM(input_size, input_size, num_layer, bidirectional=True, batch_first=True)
        self.h0 = torch.randn(2 * num_layer, batch_size, input_size).to(device)
        self.c0 = torch.randn(2 * num_layer, batch_size, input_size).to(device)
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.batch_size = batch_size
        self.hidden_size = input_size
        self.loss = nn.BCELoss()
        self.w = torch.randn(input_size).to(device)

        self.embedding_dropout = nn.Dropout(conf.dropout)
        self.lstm_dropout = nn.Dropout(conf.dropout)
        self.attention_dropout = nn.Dropout(conf.dropout)

        self.fc = nn.Sequential(nn.Linear(input_size, label_size))

    def Att_layer(self, H):
        M = self.tanh(H)
        alpha = self.softmax(torch.bmm(M, self.w.repeat(self.batch_size_var, 1, 1).transpose(1, 2)))
        res = self.tanh(torch.bmm(alpha.transpose(1,2), H))
        return res

    '''
        修复数据集中最后的非规整数据问题
        因为数据集小没有做更简单的drop处理
    '''
    def get_hc0(self, size):
        self.batch_size_var = size
        if size==self.batch_size:
            return (self.h0, self.c0)

        if self.h0_var!=None:
            del self.h0_var
            del self.c0_var
        self.h0_var = torch.randn(2 * self.num_layer, size, self.input_size).to(device)
        self.c0_var = torch.randn(2 * self.num_layer, size, self.input_size).to(device)
        return (self.h0_var, self.c0_var)

    def forward(self, x_input):
        x_input = self.embedding_dropout(x_input)

        batch_size = x_input.shape[0]
        h, _ = self.blstm(x_input, self.get_hc0(batch_size))
        h = h[:,:,self.hidden_size:] + h[:,:,:self.hidden_size]
        h = self.lstm_dropout(h)

        # print(h.shape)
        atth = h[:,-1:,:]
        # atth = self.Att_layer(h)
        # atth = self.attention_dropout(atth)
        # print(atth.shape)
        # print(h[:,-1:,:].shape)
        # exit(0)

        out = self.fc(atth)
        out = self.softmax(out)

        return out.view(self.batch_size_var, -1)

class SCVulBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)

        self.dropout = nn.Dropout(conf.dropout)
        self.classifier = nn.Linear(768, 2)
        self.blstm_att  = AttBLSTM(input_size=768, label_size=2, num_layer=1)

        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        # cls_vectors = bert_output[0][:, 0, :] # [16, len(tokens), 768]=>[16, 768]
        # cls_vectors = self.dropout(cls_vectors)
        # logits = self.classifier(cls_vectors)
        # logits = F.softmax(logits, dim=-1) # default dim=-1 for each result not batch normalization

        # bert_output[0]: [batch_size, seq_lens, 768]
        logits = self.blstm_att(bert_output[0])

        return logits

if __name__ == "__main__":
    from E55_Dataset import SCVulData
    from E56_DatasetLoader import *

    train_data        = SCVulData(conf.dataset)
    train_data_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)

    config = AutoConfig.from_pretrained(checkpoint)
    model  = SCVulBert.from_pretrained(checkpoint, config=config).to(device)

    batch_X, batch_y = next(iter(train_data_loader))
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    outputs = model(batch_X)
    print(outputs)
    print(outputs.shape)