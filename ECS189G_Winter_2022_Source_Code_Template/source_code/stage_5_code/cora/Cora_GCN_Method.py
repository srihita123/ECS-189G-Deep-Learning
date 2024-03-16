import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pygcn import GraphConvolution
from source_code.base_class.method import method
from stage_5_code.cora.Evaluate_Accuracy import Evaluate_Accuracy


class Cora_GCN_Method(nn.Module, method):
    data = None
    # Hyperparameters
    hidden1 = 100
    hidden2 = 100
    dropout = 0.5
    weight_decay = 5e-4
    learning_rate = 1e-3
    max_epoch = 80

    def __init__(self, mName, mDescription, nfeat, nclass):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.gc1 = GraphConvolution(nfeat, self.hidden1)
        self.activation_func_1 = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
        self.gc2 = GraphConvolution(self.hidden1, self.hidden2)
        self.activation_func_2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden2, nclass)
        self.output_func = nn.LogSoftmax(dim=1)

    def forward(self, x, adj):
        h1 = self.activation_func_1(self.gc1(x, adj))
        h1 = self.dropout_layer(h1)
        h2 = self.activation_func_2(self.gc2(h1, adj))
        h2 = self.dropout_layer(h2)
        # h3 = self.activation_func_3(self.fc3(h2))
        # h3 = self.dropout_layer(h3)
        h3 = self.output_func(self.fc3(h2))
        return h3

    def train_loop(self, features, labels, adj, idx):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = torch.nn.NLLLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []

        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # Forward pass on entire graph
            output = self.forward(features, adj)
            # calculate the loss ONLY USING TRAINING LABELS
            train_loss = loss_function(output[idx], labels[idx])
            # record loss for training convergence plot
            losses.append(train_loss.item())
            # gradient descent and backprop
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                # calculate accuracy on training nodes
                accuracy_evaluator.data = {'true_y': labels[idx], 'pred_y': output[idx].max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
        # Save training convergence plot
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig('../../result/stage_5_result/cora_loss.png')

    def test(self, features, adj, idx):
        self.eval()
        with torch.no_grad():
            output = self.forward(features, adj)
        # return inference on TESTING DATA only
        return output[idx].max(1)[1]

    def run(self):
        # Assuming data is set
        features = self.data['graph']['X']
        labels = self.data['graph']['y']
        adj = self.data['graph']['utility']['A']
        idx_train = self.data['train_test_val']['idx_train']
        idx_test = self.data['train_test_val']['idx_test']
        print('method running...')
        print('--start training...')
        self.train_loop(features, labels, adj, idx_train)
        print('--start testing...')
        output = self.test(features, adj, idx_test)
        return {'pred_y': output, 'true_y': labels[idx_test]}
