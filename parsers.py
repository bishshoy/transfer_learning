import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--print-model', action='store_true')

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--continuous', type=int, default=5)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=5e-4)

    parser.add_argument('--cifar', type=int)
    parser.add_argument('--imagenet', action='store_true')

    parser.add_argument('--validate', action='store_true')

    parser.add_argument('--freeze-conv', action='store_true')
    parser.add_argument('--replace-fc', action='store_true')

    parser.add_argument('--upscale-images', action='store_true')

    return parser.parse_args()
