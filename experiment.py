import torch
import torch.nn as nn

from torchmetrics import MaxMetric
from icecream import ic

from models import *
from dataloaders import *
from trainer import *
from parsers import *


def experiment(args):
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name(0))

    model = build_model(args)
    model.cuda()

    train_loader, val_loader = dataloaders(args)  # This also sets args.num_classes

    if args.freeze_conv:
        freeze_conv_layers(model, args)
    replace_fc_layer(model, args)

    if args.print_model:
        print(model)
        for param in model.parameters():
            print(param.shape, param.requires_grad)
        exit()

    model = nn.DataParallel(model)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd, nesterov=True)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    if args.validate:
        validate(model, val_loader)
        return

    best_acc = MaxMetric()
    _continuous = args.continuous
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, loss_fn, optim, train_loader, epoch, args.epochs)
        if _continuous == 1 or epoch == args.epochs - 1:
            validate(model, val_loader, best_acc)
            print()
            _continuous = args.continuous + 1
        _continuous -= 1
        scheduler.step()

    print('### model: {model}, dataset: {dataset}, lr: {lr}, best_acc: {best_acc:.2f}'
          ''.format(
              model=args.model,
              dataset=args.dataset,
              lr=args.lr,
              best_acc=100*best_acc.compute(),
          ))


if __name__ == '__main__':
    args = parse()
    ic(vars(args))

    experiment(args)