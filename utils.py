import torch
from collections import OrderedDict


def save_checkpoint(model, args):
    path = 'checkpoints/'

    if not args.pretrained:
        mode = '0'
    else:
        if args.freeze_conv:
            mode = '1'
        else:
            mode = '2'

    filename = [args.model, args.dataset, str(args.lr), mode]
    filename = '-'.join(filename) + '.ckpt'

    print('Saved', end='')
    torch.save(model.state_dict(), path + filename)


def restore_checkpoint(model, args):
    state_dict = torch.load(args.resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
