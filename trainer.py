import torchmetrics
import datetime
import time

from utils import *


def train_one_epoch(model, loss_fn, optim, train_loader, epoch, total_epochs):
    model.train()

    avg_loss = torchmetrics.MeanMetric()
    acc = torchmetrics.Accuracy()

    ips = torchmetrics.MeanMetric()

    for i, (data, target) in enumerate(train_loader):
        tic = time.time()

        data, target = data.cuda(), target.cuda()

        optim.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optim.step()

        avg_loss(loss.item())
        acc(output.cpu(), target.cpu())

        toc = time.time()

        if i != 0:
            ips(len(data) / (toc - tic))
            eta = (total_epochs - epoch) * len(train_loader) * len(data) / ips.compute().item() / 3600.0

            print(
                '\r[ {epoch:3}: {iter_a:3}/ {iter_b:3}]  LR:{lr:7.4f}  Loss:{loss:6.3f}  IPS:{ips:5.0f}  ETA: {eta}  Acc:{acc:6.2f}'
                '    '.format(
                    epoch=epoch + 1, iter_a=i + 1, iter_b=len(train_loader),
                    lr=optim.param_groups[0]['lr'],
                    loss=avg_loss.compute(),
                    ips=ips.compute(),
                    eta=str(datetime.timedelta(hours=eta)).rsplit(':', 1)[0],
                    acc=100 * acc.compute(),),
                end='')


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
    ), end='')
