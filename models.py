import torchvision as T


def build_model(args):
    models = {
        'vgg16': T.models.vgg16,
        'vgg16_bn': T.models.vgg16_bn,
        'resnet18': T.models.resnet18,
    }

    return models[args.model](pretrained=args.pretrained)
