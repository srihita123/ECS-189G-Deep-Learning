'''import pickle
import torch
import torchvision.transforms as transforms
from code.base_class.dataset import dataset

class MNIST_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading MNIST dataset...')
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            mnist_data = pickle.load(f)

        for instance in mnist_data['train']:
            X_train.append(instance['image'])
            y_train.append(instance['label'])
        for instance in mnist_data['test']:
            X_test.append(instance['image'])
            y_test.append(instance['label'])

        # Define transformations to be applied to each image
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image or numpy array to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1]
        ])

        # Apply transformations to each image
        X_train = [transform(image) for image in X_train]
        X_test = [transform(image) for image in X_test]

        # Convert lists to PyTorch tensors
        X_train = torch.stack(X_train)
        y_train = torch.tensor(y_train)
        X_test = torch.stack(X_test)
        y_test = torch.tensor(y_test)

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}'''


# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import os

class MNIST_Dataset_Loader(dataset):
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
        #print(os.path.abspath(self.dataset_source_folder_path))
        #f = open(os.path.abspath(self.dataset_source_folder_path) + "\\" + self.dataset_source_file_name, 'rb')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        for instance in data['train']:
            X_train.append(instance['image'])
            y_train.append(instance['label'])
        for instance in data['test']:
            X_test.append(instance['image'])
            y_test.append(instance['label'])

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}




