import PIL
import numpy as np
import torch
import os
from pathlib import Path
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image


def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST()
    elif name == 'FashionMNIST':
        return get_FashionMNIST()
    elif name == 'SVHN':
        return get_SVHN()
    elif name == 'CIFAR10':
        return get_CIFAR10()


def get_MNIST():
    raw_tr = datasets.MNIST('./MNIST', train=True, download=True)
    raw_te = datasets.MNIST('./MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te


def get_FashionMNIST():
    raw_tr = datasets.FashionMNIST('./FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST('./FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te


def get_SVHN():
    data_tr = datasets.SVHN('./SVHN', split='train', download=True)
    data_te = datasets.SVHN('./SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te


def get_CIFAR10():
    data_tr = datasets.CIFAR10('./CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('./CIFAR10', train=False, download=True)
    X_tr = data_tr.train_data
    Y_tr = torch.from_numpy(np.array(data_tr.train_labels))
    X_te = data_te.test_data
    Y_te = torch.from_numpy(np.array(data_te.test_labels))
    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name == 'MNIST':
        return DataHandler1
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class FastaiDataHandler(Dataset):
    def __init__(self, train_dir, test_dir, transform=None):
        assert os.path.isdir(train_dir), str("labeled_train_dir is not a dir")
        assert os.path.isdir(test_dir), str("labeled_test_dir is not a dir")
        self.transform = transform
        self.X = []
        self.Y = []
        for idx, class_dir in enumerate(os.listdir(train_dir)):
            class_files = os.listdir(str(Path(train_dir, class_dir)))
            self.X += [str(Path(train_dir, class_dir, img_file)) for img_file in class_files]
            self.Y += [idx for i in class_files]
            assert (len(self.X) == len(self.Y))
            assert (len(self.X) > 0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        print(self.X[index])

        x = PIL.Image.open(self.X[index]).convert("RGB")
        x = np.asarray(x)
        y = self.Y[index]
        print(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y, index
