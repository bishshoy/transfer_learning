import torch.nn as nn
from torchvision import models


def build_model(args):
    model_library = {
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'inceptionv3': models.inception_v3,
        'mobilenetv2': models.mobilenet_v2,
    }

    return model_library[args.model](pretrained=args.pretrained)


def freeze_conv_layers(model, args):
    for param in model.parameters():
        param.requires_grad = False

    for name in ['vgg', 'densenet', 'mobilenet']:
        if args.model[: len(name)] == name:
            for param in model.classifier.parameters():
                param.requires_grad = True

    for name in ['resnet', 'inception']:
        if args.model[: len(name)] == name:
            for param in model.fc.parameters():
                param.required_grad = True


def replace_fc_layer(model, args):
    for name in ['vgg', 'densenet', 'mobilenet']:
        if args.model[: len(name)] == name:
            if name == 'vgg':
                model.classifier[6] = nn.Linear(
                    in_features=model.classifier[6].in_features, out_features=args.num_classes, bias=True
                )
            elif name == 'densenet':
                model.classifier = nn.Linear(
                    in_features=model.classifier.in_features, out_features=args.num_classes, bias=True
                )
            if name == 'mobilenet':
                model.classifier[1] = nn.Linear(
                    in_features=model.classifier[1].in_features, out_features=args.num_classes, bias=True
                )

    for name in ['resnet', 'inception']:
        if args.model[: len(name)] == name:
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_classes, bias=True)

    model.to('cuda')
