from src_code.base_class.method import method
import torch
from torch import nn
import numpy as np
from src_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt


class Method_CNN_MNIST(method, nn.Module):
    data = None
    max_epoch = 50
    learning_rate = 2e-3
    momentum = 0.9
    mean = 0.5
    std = 0.5

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # input image is 28x28x1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=6, stride=2)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flat = nn.Flatten()

        #self.fc3 = nn.Linear(in_features=1008, out_features=100)
        self.fc3 = nn.Linear(in_features=16 * 2 * 2, out_features=100)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(in_features=100, out_features=10)
        self.act4 = nn.Softmax(dim=1)

    def forward(self, x):
        y_pred = self.act1(self.conv1(x))
        y_pred = self.pool1(y_pred)
        y_pred = self.act2(self.conv2(y_pred))
        y_pred = self.pool2(y_pred)
        y_pred = self.flat(y_pred)
        y_pred = self.act3(self.fc3(y_pred))
        y_pred = self.act4(self.fc4(y_pred))
        return y_pred

    def normalize(self, X):
        X_rescaled = (X - X.min()) / (X.max() - X.min())
        return (X_rescaled - self.mean) / self.std

    def train(self, X, y):
        #print("Inside train method")
        X_tensor = torch.FloatTensor(np.array(X))
        #print("X_tensor shape before:", X_tensor.shape)
        # Reshape X_tensor to have shape (batch_size, channels, height, width)
        X_tensor = X_tensor.unsqueeze(1)  # Add channel dimension
        #print("X_tensor shape after adding channel dimension:", X_tensor.shape)
        X_tensor = self.normalize(X_tensor)
        #print("X_tensor shape after normalization:", X_tensor.shape)
        y_true = torch.LongTensor(np.array(y))

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        losses = []

        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_true)
            losses.append(train_loss.item())
            train_loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig('../../result/stage_3_result/mnist_loss.png')

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X))
        #X_tensor = torch.permute(X_tensor, (0, 3, 1, 2))
        X_tensor = X_tensor.unsqueeze(1)
        X_tensor = self.normalize(X_tensor)
        y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1].tolist()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
