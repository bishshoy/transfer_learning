import torch.nn as nn


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
                                        out_features=args.num_classes, bias=True)
    elif args.model == 'resnet18':
        model.fc = nn.Linear(in_features=model.fc.in_features,
                             out_features=args.num_classes, bias=True)

    model.to('cuda')
