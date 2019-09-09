import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import os
import platform
from dataset import *


class Configuration(object):

    def __init__(self):
        self.load_model = True
        self.save_model = True
        self.num_workers = 0
        self.batch_size = 64
        self.image_size = 64
        self.num_gpus = 1
        self.num_to_request = 4
        self.num_to_predict = 3
        self.labeled_training_dir = 'Z:\\Data\\Images\\TrainAB'
        self.labeled_test_dir = 'Z:\\Data\\Images\\TestAB'
        self.unlabeled_dir = 'Z:\\Data\\Images\\AB'
        self.requested_dir = 'Z:\\Data\\Images\\ABRequest'
        self.predicted_dir = 'Z:\\Data\\Images\\ABPredicted'
        self.dataset = FastaiDataHandler(self.labeled_test_dir, self.labeled_training_dir)
        self.net = 'Net4'
        self.epochs = 50
        self.optimizer_args = {'lr': 0.01, 'momentum': 0.5}
        self.num_new_requests = 10
        self.model_file = 'model.pth'
# DATA_NAME = 'FashionMNIST'
# DATA_NAME = 'SVHN'
# DATA_NAME = 'CIFAR10'

    def get_num_workers(self):
        return self.num_workers


class WindowsConfig(Configuration):

    def __init__(self):
        super().__init__()
        self.num_workers = 0


def load_config()->Configuration:
    configuration: Configuration
    platform_name = platform.system()
    if platform_name is 'Windows':
        configuration = WindowsConfig()
    elif platform_name is 'Darwin':
        pass
    elif platform_name is 'Linux':
        pass
    else:
        print("Unrecognized operating system", str(platform_name), "...Using default configuration")
        configuration = Configuration()
    return configuration


def test_config():
    configuration = load_config()
    assert os.path.isdir(configuration.labeled_train_dir_dir), str("labeled_train_dir is not a dir")
    assert os.path.isdir(configuration.labeled_test_dir_dir), str("labeled_test_dir is not a dir")
    assert os.path.isdir(configuration.unlabeled_dir), str("unlabeled_dir is not a dir")
    assert os.path.isdir(configuration.requested_dir), str("requested_dir is not a dir")
    assert os.path.isdir(configuration.predicted_dir), str("predicted_dir is not a dir")


if __name__ == "__main__":
    test_config()
