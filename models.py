from torchvision import models


def build_model(args):
    models = {
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'resnet18': models.resnet18,
    }

    return models[args.model](pretrained=args.pretrained)
