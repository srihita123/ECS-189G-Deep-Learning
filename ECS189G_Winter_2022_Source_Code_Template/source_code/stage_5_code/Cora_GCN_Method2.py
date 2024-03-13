import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pygcn import GraphConvolution
from source_code.base_class.method import method
from source_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy


class Cora_GCN_Method(nn.Module, method):
    data = None
    # Hyperparameters
    hidden1 = 300
    hidden2 = 200
    dropout = 0.0
    learning_rate = 2e-3
    max_epoch = 60

    def __init__(self, mName, mDescription, nfeat, nclass):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.gc1 = GraphConvolution(nfeat, self.hidden1)
        self.activation_func_1 = nn.ReLU()
        self.gc2 = GraphConvolution(self.hidden1, self.hidden2)
        self.activation_func_2 = nn.ReLU()
        self.gc3 = GraphConvolution(self.hidden2, nclass)
        self.activation_func_3 = nn.Softmax(dim=1)

    def forward(self, x, adj):
        h1 = self.activation_func_1(self.gc1(x, adj))
        #h1 = nn.functional.dropout(h1, self.dropout, training=self.training)
        h2 = self.activation_func_2(self.gc2(h1, adj))
        h3 = self.activation_func_3(self.gc3(h2, adj))
        return h3

    def train(self, X, y, adj):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(X, adj)
            # calculate the training loss
            train_loss = loss_function(y_pred, y)
            # record loss for training convergence plot
            losses.append(train_loss.item())
            # gradient descent and backprop
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                accuracy_evaluator.data = {'true_y': y, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
        # Save training convergence plot
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig('../../result/stage_5_result/cora_loss.png')

    def test(self, X, adj):
        y_pred = self.forward(X, adj)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'], self.data['train']['adj'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'], self.data['test']['adj'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
