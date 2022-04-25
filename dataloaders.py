from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_dataloaders(args):
    if args.dataset == 'imagenet':
        return get_imagenetloaders(args)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        return get_cifarloaders(args)
    elif args.dataset == 'stl10':
        return get_stlloaders(args)


def get_imagenetloaders(args):
    args.num_classes = 1000

    train_dataset = datasets.ImageNet(args.root, split='train',
                                      transform=transforms.Compose([
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225)),
                                      ]))
    val_dataset = datasets.ImageNet(args.root, split='val',
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225)),
                                    ]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader


def get_cifarloaders(args):
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10
        args.num_classes = 10
    else:
        dataset = datasets.CIFAR100
        args.num_classes = 100

    normalize = transforms.Normalize((0.50707516, 0.48654887, 0.44091784),
                                     (0.26733429, 0.25643846, 0.27615047))

    if args.upscale_images:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.RandomCrop(240, 16),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = dataset(args.root, train=True, transform=train_transforms, download=True)

    val_dataset = dataset(args.root, train=False, transform=val_transforms, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader


def get_stlloaders(args):
    args.num_classes = 10

    normalize = transforms.Normalize((0.44671097, 0.4398105, 0.4066468),
                                     (0.2603405, 0.25657743, 0.27126738))

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96, 12),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.STL10(args.root, split='train', transform=train_transforms, download=True)

    val_dataset = datasets.STL10(args.root, split='test', transform=val_transforms, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader
