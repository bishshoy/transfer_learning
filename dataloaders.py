from torchvision import transforms
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_imagenetloaders(args):
    train_dataset = ImageNet('/imagenet', split='train',
                             transform=transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                                (0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)),
                             ]))
    val_dataset = ImageNet('/imagenet', split='val',
                           transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                              (0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225)),
                           ]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=14, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=14, pin_memory=True)

    return train_loader, val_loader


def get_cifarloaders(args):

    if args.cifar == 10:
        dataset = CIFAR10
        root = '/cifar10'
    else:
        dataset = CIFAR100
        root = '/cifar100'

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

    train_dataset = dataset(root, train=True, transform=train_transforms, download=False)
    val_dataset = dataset(root, train=False, transform=val_transforms, download=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=14, pin_memory=True, persistent_workers=True,
                              sampler=DistributedSampler(train_dataset))

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=14, pin_memory=True, persistent_workers=True,
                            sampler=DistributedSampler(val_dataset))

    return train_loader, val_loader
