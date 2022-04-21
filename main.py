import torch
import torch.nn as nn
import torchmetrics
from icecream import ic

from models import *
from dataloaders import *
from trainer import *
from utils import *
from parsers import *


def main(args):
    model = build_model(args)
    model.cuda()

    if args.freeze_conv:
        freeze_conv_layers(model, args)
    if args.replace_fc:
        replace_fc_layer(model, args)

    if args.print_model:
        print(model)
        for param in model.parameters():
            print(param.shape, param.requires_grad)
        exit()

    model = torch.nn.DataParallel(model)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd, nesterov=True)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    if args.imagenet:
        train_loader, val_loader = get_imagenetloaders(args)
    else:
        train_loader, val_loader = get_cifarloaders(args)

    if args.validate:
        validate(model, val_loader)
        return

    best_acc = torchmetrics.MaxMetric()
    _continuous = args.continuous
    for epoch in range(args.epochs):
        train_one_epoch(model, loss_fn, optim, train_loader, epoch)
        if _continuous == 1 or epoch == args.epochs - 1:
            validate(model, val_loader, best_acc)
            _continuous = args.continuous + 1
        _continuous -= 1
        scheduler.step()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = get_args()
    ic(vars(args))
    main(args)
