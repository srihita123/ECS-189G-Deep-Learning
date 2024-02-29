'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src_code.base_class.method import method
from src_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


class Method_CNN_ORL(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 2e-3
    # defines momentum hyperparameter
    momentum = 0.9
    # flag to set if passing mps arg at initialization
    mps_device = None
    # mean and standard deviation for normalizing data
    mean = 0.5
    std = 0.5

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, mps=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # input image is 112 x 92 x 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=6, stride=2)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv2d(in_channels=48, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(in_features=756, out_features=100)
        self.act3 = nn.Tanh()

        self.fc4 = nn.Linear(in_features=100, out_features=40)
        self.act4 = nn.Softmax(dim=1)
        # Set mps device if using Mac Chip
        if mps:
            if not torch.backends.mps.is_available():
                print("Could not find MPS! Using CPU instead.")
            else:
                self.mps_device = torch.device("mps")

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        y_pred = self.act1(self.conv1(x))
        y_pred = self.pool1(y_pred)
        y_pred = self.act2(self.conv2(y_pred))
        y_pred = self.pool2(y_pred)
        y_pred = self.flat(y_pred)
        y_pred = self.act3(self.fc3(y_pred))
        y_pred = self.act4(self.fc4(y_pred))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here
    def normalize(self, X : torch.Tensor):
        X_rescaled = (X - X.min()) / (X.max() - X.min())
        return (X_rescaled - self.mean) / self.std

    def train(self, X, y):

        X_tensor = torch.FloatTensor(np.array(X))
        # input should have format (batch size, # channels, height, width) where
        # batch size = 360, # channels = 1, height = 112, width = 92
        X_tensor = torch.permute(X_tensor, (3, 0, 1, 2))[0, :, :, :].unsqueeze(1)
        # min-max normalize + normalize to mean and std
        X_tensor = self.normalize(X_tensor)

        # convert class labels 1-40 to 0-39 for Cross Entropy calculation
        y_true = torch.LongTensor(np.array(y) - 1)
        # if mps_device is set, move tensors to mps
        if self.mps_device:
            X_tensor = X_tensor.to(self.mps_device)
            y_true = y_true.to(self.mps_device)

        # List to store loss values during training
        losses = []
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # perform forward propagation
            # Add extra dimension at index 1 (for 1 input channel)
            y_pred = self.forward(X_tensor)
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)
            # record loss for training convergence plot
            losses.append(train_loss.item())
            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch % 2 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
        # Save training convergence plot
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig('../../result/stage_3_result/orl_loss.png')

    def test(self, X):
        # do the testing, and result the result
        X_tensor = torch.FloatTensor(np.array(X))
        # input should have format (batch size, # channels, height, width) where
        # batch size = 40, # channels = 1, height = 112, width = 92
        X_tensor = torch.permute(X_tensor, (3, 0, 1, 2))[0, :, :, :].unsqueeze(1)
        # min-max normalize + normalize to mean and std
        X_tensor = self.normalize(X_tensor)
        # move to MPS if device is set
        if self.mps_device:
            X_tensor.to(self.mps_device)
        # feed testing data to forward propagation
        y_pred = self.forward(X_tensor)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        # NOTE: should +1 somewhere to convert back to 1-40 label indexing
        return y_pred.max(1)[1] + 1

    def run(self):
        print('method running...')
        if self.mps_device:
            print('Using MPS')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        # remember to use 0-39 indexing!
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y.tolist(), 'true_y': self.data['test']['y']}
