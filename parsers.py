import argparse
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--print-model', action='store_true')

    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--continuous', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=5e-4)

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--root', type=str, default='datasets')

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--validate', action='store_true')

    # Transfer Learning args
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--freeze-conv', action='store_true')

    args, _ = parser.parse_known_args()
    return args
