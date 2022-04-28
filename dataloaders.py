import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
import glob


def dataloaders(args):
    if args.dataset == 'imagenet':
        args.num_classes = 1000
        train_dataset, val_dataset = imagenet(args)

    elif args.dataset == 'cifar10':
        args.num_classes = 10
        train_dataset, val_dataset = cifar10(args)

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        train_dataset, val_dataset = cifar100(args)

    elif args.dataset == 'stl10':
        args.num_classes = 10
        train_dataset, val_dataset = stl10(args)

    elif args.dataset == 'caltech101':
        args.num_classes = 101
        train_dataset, val_dataset = caltech101(args)

    elif args.dataset == 'dtd':
        args.num_classes = 47
        train_dataset, val_dataset = dtd(args)

    elif args.dataset == 'sun':
        args.num_classes = 397
        train_dataset, val_dataset = sun(args)

    elif args.dataset == 'cars':
        args.num_classes = 196
        train_dataset, val_dataset = cars(args)

    elif args.dataset == 'aircraft':
        args.num_classes = 102
        train_dataset, val_dataset = aircraft(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader


def imagenet_normalizer():
    return transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))


def train_transform():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        imagenet_normalizer()
    ])

    return train_transform


def val_transform():
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        imagenet_normalizer(),
    ])

    return val_transform


def imagenet(args):
    train_dataset = datasets.ImageNet(args.root, split='train',
                                      transform=train_transform())
    val_dataset = datasets.ImageNet(args.root, split='val',
                                    transform=val_transform())

    return train_dataset, val_dataset


def cifar10(args):
    train_dataset = datasets.CIFAR10(args.root, train=True,
                                     transform=train_transform(), download=True)

    val_dataset = datasets.CIFAR10(args.root, train=False,
                                   transform=val_transform(), download=True)

    return train_dataset, val_dataset


def cifar100(args):
    train_dataset = datasets.CIFAR100(args.root, train=True,
                                      transform=train_transform(), download=True)

    val_dataset = datasets.CIFAR100(args.root, train=False,
                                    transform=val_transform(), download=True)

    return train_dataset, val_dataset


def stl10(args):
    train_dataset = datasets.STL10(args.root, split='train',
                                   transform=train_transform(), download=True)

    val_dataset = datasets.STL10(args.root, split='test',
                                 transform=val_transform(), download=True)

    return train_dataset, val_dataset


def caltech101(args):
    class Caltech101(Dataset):
        def __init__(self):
            folders = glob.glob(args.root+'/caltech101/*')

            self.files = []
            for class_idx, folder in enumerate(folders):
                files = glob.glob(folder+'/*.jpg')
                # print(len(files))

                for file in files:
                    self.files.append((file, class_idx))
            # exit()

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            file, label = self.files[idx]
            image = Image.open(file)

            # Random Horizontal Flip
            rnd = np.random.randint(0, 2, 1)[0]
            if rnd:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Resize and Crop
            image = image.resize((240, 240))
            x, y = np.random.randint(0, 16, 2)
            image = image.crop((x, y, x+224, y+224))

            # To tensor
            image = np.asarray(image)
            image = torch.from_numpy(image)

            if len(image.shape) == 2:
                image = torch.stack((image, image, image), dim=-1)

            # Normalize
            mean, stdev = torch.asarray([[[0.485, 0.456, 0.406]]]), torch.asarray([[[0.229, 0.224, 0.225]]])
            image = (image/255 - mean) / stdev

            # Permute
            image = image.permute(2, 0, 1)

            return image, label

    dataset = Caltech101()

    val_len = int(0.1 * len(dataset))
    train_len = len(dataset) - val_len

    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    return train_dataset, val_dataset


def dtd(args):
    train_dataset = datasets.DTD(args.root, split='train',
                                 transform=train_transform(), download=True)

    val_dataset = datasets.DTD(args.root, split='test',
                               transform=val_transform(), download=True)

    return train_dataset, val_dataset


def sun(args):
    class SUN397(Dataset):
        def __init__(self):
            folders = glob.glob(args.root+'/SUN397/*')

            self.files = []
            for class_idx, folder in enumerate(folders):
                files = glob.glob(folder+'/*/*.jpg')
                # print(len(files))

                for file in files:
                    self.files.append((file, class_idx))
            # exit()

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            file, label = self.files[idx]
            image = Image.open(file)

            # Random Horizontal Flip
            rnd = np.random.randint(0, 2, 1)[0]
            if rnd:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Resize and Crop
            image = image.resize((240, 240))
            x, y = np.random.randint(0, 16, 2)
            image = image.crop((x, y, x+224, y+224))

            # To tensor
            image = np.asarray(image)
            image = torch.from_numpy(image)

            if len(image.shape) == 2:
                image = torch.stack((image, image, image), dim=-1)
            if image.shape[2] == 4:
                image = image[:, :, :3]

            # Normalize
            mean, stdev = torch.asarray([[[0.485, 0.456, 0.406]]]), torch.asarray([[[0.229, 0.224, 0.225]]])
            image = (image/255 - mean) / stdev

            # Permute
            image = image.permute(2, 0, 1)

            return image, label

    dataset = SUN397()

    val_len = int(0.1 * len(dataset))
    train_len = len(dataset) - val_len

    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    return train_dataset, val_dataset


def cars(args):
    train_dataset = datasets.StanfordCars(args.root, split='train',
                                          transform=train_transform(), download=True)

    val_dataset = datasets.StanfordCars(args.root, split='test',
                                        transform=val_transform(), download=True)

    return train_dataset, val_dataset


def aircraft(args):
    train_dataset = datasets.FGVCAircraft(args.root, split='train',
                                          transform=train_transform(), download=True)

    val_dataset = datasets.FGVCAircraft(args.root, split='test',
                                        transform=val_transform(), download=True)

    return train_dataset, val_dataset
