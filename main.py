import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from icecream import ic
import random
import os

from models import *
from dataloaders import *
from trainer import *
from utils import *
from parsers import *


def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device('cuda:'+str(args.rank))
    torch.cuda.manual_seed_all(random.randint(1e5, 1e6))
    args.world_size = dist.get_world_size()

    model = build_model(args)

    if args.freeze_conv:
        freeze_conv_layers(model, args)
    if args.replace_fc:
        replace_fc_layer(model, args)

    if args.print_model:
        pprint(model)
        for param in model.parameters():
            pprint(param.shape, param.requires_grad)
        exit()

    model = DDP(model.cuda(), device_ids=[args.rank], output_device=args.rank)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    if args.imagenet:
        train_loader, val_loader = get_imagenetloaders(args)
    else:
        train_loader, val_loader = get_cifarloaders(args)

    best_acc = Max()

    if args.validate:
        validate(model, val_loader, best_acc, args)
        return

    _continuous = args.continuous

    for epoch in range(args.epochs):
        train_one_epoch(model, loss_fn, optim, train_loader, epoch, args)

        if _continuous == 1 or epoch == args.epochs - 1:
            validate(model, val_loader, best_acc, args)
            pprint()
            _continuous = args.continuous + 1

        _continuous -= 1
        scheduler.step()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    args = get_args()
    args.rank = int(os.environ['LOCAL_RANK'])

    if args.rank == 0:
        ic(vars(args))

    main(args)
