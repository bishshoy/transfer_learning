from torchvision import models


def build_model(args):
    model_library = {
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'resnet18': models.resnet18,
    }

    return model_library[args.model](pretrained=args.pretrained)
