'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from source_code.base_class.method import method
from source_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split
from tqdm import tqdm


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 2e-3
    # defines momentum hyperparameter
    momentum = 0.9
    # flag to set if passing mps arg at initialization
    mps_device = None
    # mean and standard deviation for normalizing data
    mean = 0.5
    std = 0.5

    def __init__(self, mName, mDescription, mps=False):
        method.__init__(self, mName, mDescription)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        nn.Module.__init__(self)
        # input image is 32 x 32 x 3
        # output size = (n + 2p - f)/s + 1

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # output: 64 x 16 x 16
        self.norm1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # output: 128 x 8 x 8
        self.norm2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # output: 256 x 4 x 4
        self.norm3 = nn.BatchNorm2d(256)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.act7 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.act8 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)

        self.to(self.device)

        # self.act4 = nn.Softmax(dim=1)
        # Set mps device if using Mac Chip
        if mps:
            if not torch.backends.mps.is_available():
                print("Could not find MPS! Using CPU instead.")
            else:
                self.mps_device = torch.device("mps")

    def forward(self, x):
        '''Forward propagation'''
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)  # I assume this was intended to be after conv2 based on typical CNN patterns
        x = self.norm1(x)  # Applying BatchNorm after pooling

        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)
        x = self.norm2(x)  # Applying BatchNorm after pooling

        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.pool3(x)
        x = self.norm3(x)  # Applying BatchNorm after pooling

        x = self.flat(x)
        x = self.act7(self.fc1(x))
        x = self.act8(self.fc2(x))
        y_pred = self.fc3(x)  # No activation here as nn.CrossEntropyLoss expects raw scores

        return y_pred

    def train(self, X, y):

        print("Number of points:", len(X))
        print("Number of features:", len(X[0]), len(X[0][0]), len(X[0][0][0]))
        print("Unique values", set(y))

        # List to store loss values during training
        losses = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        if isinstance(X, list):
            X = np.array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # Normalize to [0, 1]
        y_tensor = torch.tensor(y, dtype=torch.long)

        # DataLoader setup
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        y_true = torch.LongTensor(np.array(y))

        for epoch in range(self.max_epoch):
            total_loss = 0

            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)  # Move data to the device

                optimizer.zero_grad()

                y_pred = self.forward(data)

                train_loss = loss_function(y_pred, target)
                total_loss += train_loss.item()


                train_loss.backward()
                optimizer.step()
            losses.append(total_loss)

            avg_loss = total_loss / len(loader)
            # accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            # print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
            print(f'Epoch [{epoch + 1}/{self.max_epoch}], Loss: {avg_loss:.4f}')

        # Save training convergence plot
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig('../../result/stage_3_result/cifar_loss.png')

        # return history

    def test(self, X):
        if isinstance(X, list):
            X = np.array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # Normalize to [0, 1]
        X_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(X_tensor)

        test_dataset = TensorDataset(X_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Adjust batch_size as needed

        all_predictions = []
        with torch.no_grad():
            for data in test_loader:
                data = data[0].to(self.device)  # Assuming there are no labels in test_loader
                output = self(data)
                _, predicted = torch.max(output.data, 1)
                all_predictions.extend(predicted.to("cpu").numpy())  # Move predictions back to CPU

        return np.array(all_predictions)

        # X_tensor = X_tensor.to(self.device)
        # y_tensor = y_tensor.to(self.device)

        # if self.mps_device:
        #     X_tensor.to(self.mps_device)
        # # feed testing data to forward propagation
        # y_pred = self.forward(X_tensor)
        # return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        if self.mps_device:
            print('Using MPS')
        print('--start training...')

        # val_size = 5000
        # train_size = len(self.data['train']['X']) - val_size
        # train_ds, val_ds = random_split(self.data['train']['X'], [train_size, val_size])
        #
        # batch_size = 64
        # train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
        # history = self.train(train_dl, val_dl)

        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        # remember to use 0-39 indexing!
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y.tolist(), 'true_y': self.data['test']['y']}
