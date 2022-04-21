import torch.distributed as dist
import datetime
import time

from utils import *


def train_one_epoch(model, loss_fn, optim, train_loader, epoch, args):
    dist.barrier()
    model.train()

    avg_loss = Average()
    acc = Accuracy()
    tic = time.time()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optim.zero_grad()

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optim.step()

        avg_loss(loss)
        acc(output, target)

        if i != 0:
            if i != len(train_loader) - 1:
                toc = time.time()
                ips = args.world_size * len(data) / (toc - tic)
                tic = time.time()
                eta = (args.epochs - epoch) * len(train_loader) * len(data) * args.world_size / ips / 3600.0

            final_avg_loss = avg_loss.compute().item()
            final_acc = 100*acc.compute().item()
            if args.rank == 0:
                pprint(
                    '\r[ {epoch:3}: {iter_a:3}/ {iter_b:3}]  LR:{lr:7.4f}  Loss:{loss:6.3f}  IPS:{ips:5.0f}  ETA: {eta}  Acc:{acc:6.2f}'
                    '        '.format(
                        epoch=epoch+1, iter_a=i+1, iter_b=len(train_loader),
                        lr=optim.param_groups[0]['lr'], loss=final_avg_loss,
                        ips=ips, eta=str(datetime.timedelta(hours=eta)).rsplit(':', 1)[0],
                        acc=final_acc),
                    end='')


def validate(model, val_loader, best_acc, args):
    dist.barrier()
    model.eval()

    acc = Accuracy()

    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()

        output = model(data)
        acc(output, target)

    best_acc(acc.compute())
    final_acc = 100*acc.compute().item()
    final_best_acc = 100*best_acc.compute().item()

    if args.rank == 0:
        pprint('Val: {val_acc:5.2f}   Best: {best_acc:5.2f}    '.format(
            val_acc=final_acc, best_acc=final_best_acc
        ), end='')
