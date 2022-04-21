import torch
import torchvision as T


def get_imagenetloaders(args):
    train_dataset = T.datasets.ImageNet('/imagenet', split='train',
                                        transform=T.transforms.Compose([
                                            T.transforms.RandomResizedCrop(224),
                                            T.transforms.RandomHorizontalFlip(),
                                            T.transforms.ToTensor(),
                                            T.transforms.Normalize(
                                                (0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)),
                                        ]))
    val_dataset = T.datasets.ImageNet('/imagenet', split='val',
                                      transform=T.transforms.Compose([
                                          T.transforms.Resize(256),
                                          T.transforms.CenterCrop(224),
                                          T.transforms.ToTensor(),
                                          T.transforms.Normalize(
                                              (0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225)),
                                      ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=14, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=14, pin_memory=True)

    return train_loader, val_loader


def get_cifarloaders(args):

    if args.cifar == 10:
        dataset = T.datasets.CIFAR10
        root = '/cifar10'
    else:
        dataset = T.datasets.CIFAR100
        root = '/cifar100'

    normalize = T.transforms.Normalize((0.50707516, 0.48654887, 0.44091784),
                                       (0.26733429, 0.25643846, 0.27615047))

    if args.upscale_images:
        train_transforms = T.transforms.Compose([
            T.transforms.RandomHorizontalFlip(),
            T.transforms.Resize(224),
            T.transforms.RandomCrop(240, 16),
            T.transforms.ToTensor(),
            normalize,
        ])
        val_transforms = T.transforms.Compose([
            T.transforms.Resize(224),
            T.transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transforms = T.transforms.Compose([
            T.transforms.RandomHorizontalFlip(),
            T.transforms.RandomCrop(32, 4),
            T.transforms.ToTensor(),
            normalize,
        ])
        val_transforms = T.transforms.Compose([
            T.transforms.ToTensor(),
            normalize,
        ])

    train_dataset = dataset(root, train=True, transform=train_transforms, download=False)

    val_dataset = dataset(root, train=False, transform=val_transforms, download=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=14, pin_memory=True,
        persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=14, pin_memory=True,
        persistent_workers=True)

    return train_loader, val_loader
