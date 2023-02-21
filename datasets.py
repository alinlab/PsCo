import os
import json
import random
from collections import defaultdict

import pandas as pd
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torch
import torch.nn as nn
import torchvision.datasets as D
import torchvision.transforms as T
import ignite.distributed as idist
from torchvision.datasets.utils import list_files


FEWSHOT_BENCHMARKS = ['omniglot', 'miniimagenet', 'cub200', 'cropdiseases', 'eurosat', 'isic', 'chestx', 'places', 'cars', 'plantae']


class RandomNoise(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, x):
        noise = np.random.choice([-1, 0, 1], x.shape[0], p=[self.ratio/2, 1-self.ratio, self.ratio/2])
        x = np.abs(x-noise)
        return x


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultipleTransform(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, x):
        return [t(x) for t in self.transforms]


class FewShotTaskSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, N, K, Q, num_tasks):
        self.N = N
        self.K = K
        self.Q = Q
        self.num_tasks = num_tasks

        if isinstance(dataset, (D.CIFAR10, D.CIFAR100, ISIC2018, ChestX)):
            labels = dataset.targets
        elif isinstance(dataset, (D.ImageFolder, Omniglot, JSONImageDataset, Cars)):
            labels = [y for _, y in dataset.samples]
        else:
            raise NotImplementedError

        self.indices = defaultdict(list)
        for i, y in enumerate(labels):
            self.indices[y].append(i)

    def __iter__(self):
        for _ in range(self.num_tasks):
            batch_indices = []
            labels = random.sample(list(self.indices.keys()), self.N)
            for y in labels:
                if len(self.indices[y]) >= self.K+self.Q:
                    batch_indices.extend(random.sample(self.indices[y], self.K+self.Q))
                else:
                    batch_indices.extend(random.choices(self.indices[y], k=self.K+self.Q))
            yield batch_indices


class Omniglot(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        super().__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.rot = [0, 90, 180, 270]

        with open(os.path.join(self.root, f'vinyals_{self.split}_labels.json'), mode='r') as f:
            dir_list = json.load(f)

        self.char_dirs = []
        for dir_ in dir_list:
            self.char_dirs.append(os.path.join(self.root, *dir_))

        self.character_images = [
            [
                [
                    (os.path.join(char_dir, image), idx+i*len(self.char_dirs)) for image in (list_files(char_dir, ".png")+list_files(char_dir, ".jpg"))
                ] for idx, char_dir in enumerate(self.char_dirs)
            ] for i in range(4)
        ]
        self.character_images = sum(self.character_images, [])
        self.samples = sum(self.character_images, [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, y = self.samples[index]
        filename = os.path.join(self.root, filename)

        img = Image.open(filename, mode='r').convert('L')
        img = img.rotate(self.rot[y//len(self.char_dirs)])

        return self.transform(img), y


class JSONImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, root, split, transform, depth=0):
        super().__init__()

        with open(f'splits/{dataset}/{split}.json', 'r') as f:
            dir_list = json.load(f)
        class_to_idx = {category: i for i, category in enumerate(dir_list['label_names'])}

        self.samples = []

        for path in dir_list['image_names']:
            file_path = os.path.join(root, path)
            category = path.split('/')[depth]
            self.samples.append((file_path, class_to_idx[category]))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, y = self.samples[index]
        img = Image.open(filename, mode='r').convert("RGB")
        return self.transform(img), y


class Cars(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        super().__init__()

        with open(f'splits/cars/{split}.json', 'r') as f:
            dir_list = json.load(f)

        self.samples = []

        for file_name, index in zip(dir_list['image_names'], dir_list['image_labels']):
            file_path = os.path.join(root, file_name)
            self.samples.append((file_path, int(index)))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, y = self.samples[index]
        img = Image.open(filename, mode='r').convert("RGB")
        return self.transform(img), y


class ISIC2018(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super().__init__()

        self.img_path = os.path.join(root, 'ISIC2018_Task3_Training_Input')
        target_file = os.path.join(root, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')

        self.data_info = pd.read_csv(target_file, skiprows=[0], header=None)
        self.image_names = np.asarray(self.data_info.iloc[:, 0])
        self.targets = np.asarray(self.data_info.iloc[:, 1:])
        self.targets = (self.targets != 0).argmax(axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        filename = os.path.join(self.img_path, self.image_names[index] + ".jpg")
        img = Image.open(filename, mode='r').convert("RGB")
        target = self.targets[index]
        return self.transform(img), target


class ChestX(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super().__init__()

        self.img_path = os.path.join(root, 'images')
        target_file = os.path.join(root, 'Data_Entry_2017.csv')

        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
        labels_set = []

        self.data_info = pd.read_csv(target_file, skiprows=[0], header=None)
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.targets_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_names  = []
        self.targets = []
        for name, label in zip(self.image_name_all, self.targets_all):
            label = label.split("|")
            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.targets.append(self.labels_maps[label[0]])
                self.image_names.append(name)
        self.image_names = np.asarray(self.image_names)
        self.targets = np.asarray(self.targets)

        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        filename = os.path.join(self.img_path, self.image_names[index])
        img = Image.open(filename, mode='r').convert("RGB")
        target = self.targets[index]
        return self.transform(img), target


def get_augmentation(dataset, method='none'):
    interpolation=T.InterpolationMode.BICUBIC
    if dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2023, 0.1994, 0.2010]
        if method == 'none':
            return T.Compose([T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'strong':
            return T.Compose([T.RandomResizedCrop(32, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              T.RandomGrayscale(p=0.2),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'weak':
            return T.Compose([T.RandomResizedCrop(32, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    elif dataset == 'miniimagenet':
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if method == 'none':
            return T.Compose([T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'strong':
            return T.Compose([T.RandomResizedCrop(84, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              T.RandomGrayscale(p=0.2),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'weak':
            return T.Compose([T.RandomResizedCrop(84, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    elif dataset == 'omniglot':
        mean = [0.92206]
        std  = [0.08426]
        if method == 'none':
            return T.Compose([T.Resize((28, 28)),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method in ['strong', 'weak']:
            return T.Compose([T.RandomResizedCrop(28, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    elif dataset in ['imagenet']:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if method == 'none':
            return T.Compose([T.Resize(256),
                              T.CenterCrop(224),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'strong':
            return T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              T.RandomGrayscale(p=0.2),
                              T.RandomApply([GaussianBlur()], p=0.5),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'weak':
            return T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    elif isinstance(dataset, list) and dataset[1] in ['cub200', 'cropdiseases', 'eurosat', 'isic', 'chestx', 'places', 'cars', 'plantae']:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if dataset[0] == 'imagenet':
            return T.Compose([T.Resize(256, interpolation=interpolation),
                              T.CenterCrop(224),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

        else:
            return T.Compose([T.Resize(84, interpolation=interpolation),
                              T.CenterCrop(84),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])

    else:
        raise NotImplementedError


def get_dataset(dataset, datadir, augmentations=['strong', 'strong']):
    if dataset == 'cifar10':
        augs = [get_augmentation(dataset, aug) for aug in augmentations]
        train = D.CIFAR10(datadir, train=True,  transform=MultipleTransform(augs))
        val   = D.CIFAR10(datadir, train=True,  transform=get_augmentation(dataset, 'none'))
        test  = D.CIFAR10(datadir, train=False, transform=get_augmentation(dataset, 'none'))
        num_classes = 10
        input_shape = (3, 32, 32)

    elif dataset == 'miniimagenet':
        augs = [get_augmentation(dataset, aug) for aug in augmentations]
        train = D.ImageFolder(os.path.join(datadir, 'train'), transform=MultipleTransform(augs))
        val   = D.ImageFolder(os.path.join(datadir, 'val'),   transform=get_augmentation(dataset, 'none'))
        test  = D.ImageFolder(os.path.join(datadir, 'test'),  transform=get_augmentation(dataset, 'none'))
        num_classes = (64, 16, 20)
        input_shape = (3, 84, 84)

    elif dataset == 'omniglot':
        augs = [get_augmentation(dataset, aug) for aug in augmentations]
        train = Omniglot(datadir, 'train', transform=MultipleTransform(augs))
        val   = Omniglot(datadir, 'val',   transform=get_augmentation(dataset, 'none'))
        test  = Omniglot(datadir, 'test',  transform=get_augmentation(dataset, 'none'))
        num_classes = (1200, 100, 323)
        input_shape = (1, 28, 28)

    elif dataset in ['imagenet', 'imagenet100']:
        augs = [get_augmentation(dataset, aug) for aug in augmentations]
        train = D.ImageFolder(os.path.join(datadir, 'train'), transform=MultipleTransform(augs))
        val   = D.ImageFolder(os.path.join(datadir, 'train'), transform=get_augmentation(dataset, 'none'))
        test  = D.ImageFolder(os.path.join(datadir, 'val'),   transform=get_augmentation(dataset, 'none'))
        num_classes = 1000 if dataset == 'imagenet' else 100
        input_shape = (3, 224, 224)

        if dataset == 'imagenet':
            indices = torch.load('imagenet_indices.pth')
            val = torch.utils.data.Subset(val, indices)

    elif isinstance(dataset, list): #Cross-domain
        if dataset[1] == 'cub200':
            train = JSONImageDataset('cub200', datadir, 'base',  transform=get_augmentation(dataset, 'none'), depth=1)
            val   = JSONImageDataset('cub200', datadir, 'val',   transform=get_augmentation(dataset, 'none'), depth=1)
            test  = JSONImageDataset('cub200', datadir, 'novel', transform=get_augmentation(dataset, 'none'), depth=1)
            num_classes = (100, 50, 50)

        elif dataset[1] == 'cropdiseases':
            train = D.ImageFolder(os.path.join(datadir, 'train'), transform=get_augmentation(dataset, 'none'))
            val   = D.ImageFolder(os.path.join(datadir, 'train'), transform=get_augmentation(dataset, 'none'))
            test  = D.ImageFolder(os.path.join(datadir, 'train'), transform=get_augmentation(dataset, 'none'))
            num_classes = 38

        elif dataset[1]  == 'eurosat':
            train = D.ImageFolder(datadir, transform=get_augmentation(dataset, 'none'))
            val   = D.ImageFolder(datadir, transform=get_augmentation(dataset, 'none'))
            test  = D.ImageFolder(datadir, transform=get_augmentation(dataset, 'none'))
            num_classes = 10

        elif dataset[1] == 'isic':
            train = ISIC2018(datadir, transform=get_augmentation(dataset, 'none'))
            val   = ISIC2018(datadir, transform=get_augmentation(dataset, 'none'))
            test  = ISIC2018(datadir, transform=get_augmentation(dataset, 'none'))
            num_classes = 7

        elif dataset[1] == 'chestx':
            train = ChestX(datadir, transform=get_augmentation(dataset, 'none'))
            val   = ChestX(datadir, transform=get_augmentation(dataset, 'none'))
            test  = ChestX(datadir, transform=get_augmentation(dataset, 'none'))
            num_classes = 7

        elif dataset[1] == 'places':
            train = JSONImageDataset('places', datadir, 'base',  transform=get_augmentation(dataset, 'none'), depth=1)
            val   = JSONImageDataset('places', datadir, 'val',   transform=get_augmentation(dataset, 'none'), depth=1)
            test  = JSONImageDataset('places', datadir, 'novel', transform=get_augmentation(dataset, 'none'), depth=1)
            num_classes = (183, 91, 91)

        elif dataset[1] == 'cars':
            train = Cars(datadir, 'base',  transform=get_augmentation(dataset, 'none'))
            val   = Cars(datadir, 'val',   transform=get_augmentation(dataset, 'none'))
            test  = Cars(datadir, 'novel', transform=get_augmentation(dataset, 'none'))
            num_classes = (98, 49, 49)

        elif dataset[1] == 'plantae':
            train = JSONImageDataset('plantae', datadir, 'base',  transform=get_augmentation(dataset, 'none'))
            val   = JSONImageDataset('plantae', datadir, 'val',   transform=get_augmentation(dataset, 'none'))
            test  = JSONImageDataset('plantae', datadir, 'novel', transform=get_augmentation(dataset, 'none'))
            num_classes = (100, 50, 50)

        else:
            raise Exception(f'Unkown Datset: {dataset[1]}')
        input_shape = (3, 224, 224) if dataset[0] == 'imagenet' else (3, 84, 84)

    else:
        raise Exception(f'Unknown Dataset: {dataset}')

    return dict(train=train,
                val=val,
                test=test,
                num_classes=num_classes,
                input_shape=input_shape)


def get_loader(args, dataset, splits=['train', 'val', 'test']):
    loader = {}
    loader['train'] = idist.auto_dataloader(dataset['train'],
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=True, drop_last=True,
                                            pin_memory=True)
    for split in ['val', 'test']:
        if args.dataset not in FEWSHOT_BENCHMARKS:
            loader[split] = idist.auto_dataloader(dataset[split],
                                                  batch_size=args.batch_size,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)
        else:
            batch_sampler = FewShotTaskSampler(dataset[split], N=args.N, K=args.K, Q=args.Q,
                                               num_tasks=args.num_tasks // idist.get_world_size())
            loader[split] = torch.utils.data.DataLoader(dataset[split],
                                                        batch_sampler=batch_sampler,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    return loader

