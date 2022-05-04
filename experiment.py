import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from torchmetrics import MaxMetric
from icecream import ic

from models import *
from dataloaders import *
from trainer import *
from utils import *
from parsers import *


def experiment(args):
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name(0))

    model = build_model(args)
    train_loader, val_loader = dataloaders(args)  # This also sets args.num_classes

    if args.freeze_conv:
        freeze_conv_layers(model, args)
    replace_fc_layer(model, args)

    if args.print_model:
        print(model)
        for param in model.parameters():
            print(param.shape, param.requires_grad)
        exit()

    model = nn.DataParallel(model.cuda())
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd, nesterov=True)
    scaler = GradScaler(enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    if args.resume:
        restore_checkpoint(model, args)

    best_acc = MaxMetric()

    if args.validate:
        validate(model, val_loader, best_acc, args)
        return

    _continuous = args.continuous

    for epoch in range(args.epochs):
        train_one_epoch(model, loss_fn, optim, scaler, train_loader, epoch, args)

        if _continuous == 1 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_loader, best_acc, args)

            # Save checkpoint based on best_acc
            if val_metrics['val_acc'] >= val_metrics['best_acc']:
                save_checkpoint(model, args)

            print()
            _continuous = args.continuous + 1

        _continuous -= 1
        scheduler.step()

    print(
        '### model: {model}, dataset: {dataset}, lr: {lr}, mode: {mode}, best_acc: {best_acc:.2f}'
        ''.format(
            model=args.model,
            dataset=args.dataset,
            lr=args.lr,
            mode=mode_value(args),
            best_acc=100 * best_acc.compute(),
        )
    )


if __name__ == '__main__':
    args = parse()
    ic(vars(args))

    experiment(args)
