import torchmetrics
from utils import *


def train_one_epoch(model, loss_fn, optim, train_loader, epoch):
    model.train()

    avg_loss = torchmetrics.MeanMetric()
    acc = torchmetrics.Accuracy()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optim.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optim.step()

        avg_loss(loss.item())
        acc(output.cpu(), target.cpu())

        print('\r[ {epoch:3}: {iter_a:3}/ {iter_b:3}]  LR:{lr:7.4f}  Loss:{loss:6.3f}  Acc:{acc:6.2f}'
              '    '.format(
                  epoch=epoch+1,
                  iter_a=i+1, iter_b=len(train_loader),
                  lr=optim.param_groups[0]['lr'],
                  loss=avg_loss.compute(), acc=100*acc.compute(),
              ), end='')


def validate(model, val_loader, best_acc):
    model.eval()

    acc = torchmetrics.Accuracy()

    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()

        output = model(data)
        acc(output.cpu(), target.cpu())

    best_acc(acc.compute())
    print('Val: {val_acc:5.2f}   Best: {best_acc:5.2f}    '.format(
        val_acc=100*acc.compute(), best_acc=100*best_acc.compute()
    ))
