'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from source_code.base_class.dataset import dataset
# import torchvision.transforms as transforms
# import torch
# import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        for instance in data['train']:
            X_train.append(instance['image'])
            y_train.append(instance['label'])
        for instance in data['test']:
            X_test.append(instance['image'])
            y_test.append(instance['label'])
        '''
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize(0.5, 0.5)
        ])
        X_train_t = torch.from_numpy(np.array(X_train))
        X_train_t = transform(X_train_t)
        '''
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = CIFAR10Dataset(X_train, y_train, transform=transform)
        test_dataset = CIFAR10Dataset(X_test, y_test, transform=transform)

        # Create data loaders
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return {'train_dataset': train_loader, 'test_dataset': test_loader}