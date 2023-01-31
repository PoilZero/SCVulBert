# coding:utf8
from tqdm.auto import tqdm # for iter进度条
from torch.optim import AdamW
from transformers import get_scheduler

from E60_models.E61_model import *

def train_once(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')

    finish_step_num = (epoch - 1) * len(dataloader)
    size = len(dataloader.dataset)
    total_loss, correct = 0, 0

    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_loss += loss.item()

        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)

    total_loss /= size
    correct /= size
    print(f'loss: {total_loss} accuracy: {correct}')

def test_once(dataloader, model, mode='Valid'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100 * correct):>0.1f}%\n")
    return correct

def train(train_data_loader, valid_data_loader):
    # model load
    config = AutoConfig.from_pretrained(checkpoint)
    model  = SCVulBert.from_pretrained(checkpoint, config=config).to(device)

    # train && valid
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    lr_scheduler = get_scheduler(
        'linear'
        , optimizer=optimizer
        , num_warmup_steps=0
        , num_training_steps=conf.epoch_num * len(train_data_loader)
    )

    ## epoch loop
    for t in range(conf.epoch_num):
        print(f"Epoch {t + 1}/{conf.epoch_num}\n-------------------------------")
        train_once(train_data_loader, model, loss_fn, optimizer, lr_scheduler, t + 1)
        evaluate_once(valid_data_loader, model, loss_fn, mode='Valid')

def evaluate_once(dataloader, model, loss_fn, mode='Valid'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    total_loss, correct = 0, 0
    tn, fp, fn, tp = 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            loss = loss_fn(pred, y)
            total_loss += loss.item()
            '''
            t/f:
                pred.argmax(1) == torch.max(pred, 1)[1] 即取一列，其中每个元素都是对应行的最大值 取索引
                pred.argmax(1) == y tensor([False,  True, False,  True,  True,  True,  True]) 
            p/n:
                y
                
            tf = (pred.argmax(1) == y).type(torch.int)
            pn = y
            
            -y+1 == ~y binary
            '''
            tf = (pred.argmax(1) == y).type(torch.int)

            tp += (tf * y).sum().item()
            tn += (tf*(-y+1)).sum().item()
            fp += ((-tf+1) * y).sum().item()
            fn += ((-tf+1) * (-y+1)).sum().item()

    correct /= size
    total_loss /= size

    print("== Valid Evaluate")
    print("Loss: ", total_loss)
    print(f"Accuracy: {100 * correct}")
    # print(f"{mode} Accuracy: {(100 * correct):>0.1f}%")
    print('False positive rate(FP): ', fp / (fp + tn) if (fp + tn)!=0 else 'NA')
    print('False negative rate(FN): ', fn / (fn + tp) if (fn + tp)!=0 else 'NA')
    recall = tp / (tp + fn) if (tp + fn)!=0 else 'NA'
    print('Recall: ', recall)
    precision = tp / (tp + fp) if (tp + fp)!=0 else 'NA'
    print('Precision: ', precision)
    try:
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
    except:
        print('F1 score: ', 'NA')
    print()

if __name__ == '__main__':
    # dataset load && tokenlization
    # all_data = SCVulData(sys.argv[1] if len(sys.argv)>=2 else None)
    all_data = SCVulData()

    # train_data, valid_data = random_split(all_data, [
    #     int(len(all_data) * 0.8), len(all_data) - int(len(all_data) * 0.8)
    # ])
    div = int(len(all_data) * 0.8)
    train_data, valid_data = all_data[:div], all_data[div:]

    train_data_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)
    valid_data_loader = DataLoader(valid_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)

    #train_data, valid_data = random_split(all_data, [
    #    int(len(all_data) * 0.8), len(all_data) - int(len(all_data) * 0.8)
    #])

    # train&&evaluate
    train(train_data_loader, valid_data_loader)

