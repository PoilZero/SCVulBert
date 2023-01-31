# coding:utf8
from transformers import AutoConfig
from transformers import BertPreTrainedModel, BertModel

from E50_utils.E56_DatasetLoader import *

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
        atth = self.Att_layer(h)
        atth = self.attention_dropout(atth) # [bs, seq, 768]=>[bs, 768]
        return atth.squeeze()
        # out = self.fc(atth)
        # out = self.softmax(out)
        #
        # return out.view(self.batch_size_var, -1)

class SCVulBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)

        self.dropout = nn.Dropout(conf.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, 2)
            , nn.Softmax(dim=-1) # default dim=-1 for each result not batch normalization
        )

        # self.blstm_att  = AttBLSTM(input_size=768, label_size=2, num_layer=1)
        # self.fusion = nn.Sequential(
        #     nn.Linear(768 * 2, 2)
        #     , nn.Softmax(dim=-1)
        # )

        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)

        cls_vectors = bert_output[0][:, 0, :] # [16, len(tokens), 768]=>[16, 768]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)

        # logits1 = bert_output[1]
        # logits2 = self.blstm_att(bert_output[0])
        #
        # logits = torch.cat([logits1, logits2], dim=-1)
        # logits = self.fusion(logits)

        return logits

if __name__ == "__main__":
    train_data        = SCVulData(conf.dataset)
    train_data_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)

    config = AutoConfig.from_pretrained(checkpoint)
    model  = SCVulBert.from_pretrained(checkpoint, config=config).to(device)

    batch_X, batch_y = next(iter(train_data_loader))
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    outputs = model(batch_X)
    print(outputs)
    print(outputs.shape)