import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ReduceOp, all_reduce


def pprint(*x, end):
    if dist.get_rank() == 0:
        print(*x, end=end)


def freeze_conv_layers(model, args):
    for param in model.parameters():
        param.requires_grad = False

    if args.model == 'vgg16' or args.model == 'vgg16_bn':
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif args.model == 'resnet18':
        for param in model.fc.parameters():
            param.required_grad = True


def replace_fc_layer(model, args):
    if args.model == 'vgg16' or args.model == 'vgg16_bn':
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features,
                                        out_features=args.cifar, bias=True)
    elif args.model == 'resnet18':
        model.fc = nn.Linear(in_features=model.fc.in_features,
                             out_features=args.cifar, bias=True)

    model.to('cuda')


class Average:
    def __init__(self):
        self.x = torch.zeros(1).cuda()
        self.N = torch.zeros(1).cuda()

    def __call__(self, x):
        all_reduce(x)
        x /= dist.get_world_size()
        self.x += x
        self.N += 1

    def compute(self):
        return self.x / self.N


class Max:
    def __init__(self):
        self.x = torch.zeros(1).cuda()

    def __call__(self, x):
        all_reduce(x, op=ReduceOp.MAX)
        if self.x[0] < x[0]:
            self.x[0] = x[0]

    def compute(self):
        return self.x


class Accuracy:
    def __init__(self):
        self.correct = torch.zeros(1).cuda()
        self.N = torch.zeros(1).cuda()

    def __call__(self, output, target):
        self.correct += (output.argmax(dim=-1) == target).sum()
        self.N += len(target)

    def compute(self):
        correct, N = self.correct.clone(), self.N.clone()
        all_reduce(correct)
        all_reduce(N)
        return correct / N
