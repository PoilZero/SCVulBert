# coding:utf8

from E50_com import *
from transformers import AutoConfig
from transformers import BertPreTrainedModel, BertModel

class SCVulBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(768, 2)
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)

        logits = self.classifier(cls_vectors)
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