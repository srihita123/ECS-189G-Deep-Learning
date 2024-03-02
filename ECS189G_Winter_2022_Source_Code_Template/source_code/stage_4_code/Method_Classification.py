from sklearn.metrics import classification_report
from tqdm import tqdm

from source_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from source_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

class Method_Classification(method, nn.Module):

    input_size = 1 # have to change
    hidden_size = 256
    num_layers = 2
    num_classes = 1  # Positive or Negative sentiment
    batch_size = 64
    embedding_size = 100
    learning_rate = 0.001
    max_epochs = 10
    grad_clip = 5
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': max_epochs
    }

    losses = []
    accuracies = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        super(Method_Classification, self).__init__()
        self.optimizer = None
        self.criterion = None
        self.input_size = 100
        self.vocab_size = 69966

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)  # Fully connected layer
        self.sig = nn.Sigmoid()

    def initialize_embeddings(self, embedding_matrix):
        """Initialize the embedding layer with preloaded weights."""
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        x = x.long()

        x_embedded = self.embedding(x)
        # print("embedding", x_embedded.shape ,x_embedded)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hidden = self.lstm(x_embedded)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        out = self.sig(out)
        return out

    def train_model(self, train_loader):
        self.train()  # Set the model to training mode

        print("train_loader size (in train_model)", len(train_loader))
        for epoch in range(self.max_epochs):

            train_loss = 0
            train_acc = 0
            total_loss = 0
            print(f'\nEpoch {epoch + 1}/{self.max_epochs}')
            for id, (feature, target) in enumerate(train_loader):
                # Forward pass

                self.optimizer.zero_grad()

                out = self.forward(feature)

                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5])
                equals = predicted == target
                acc = torch.mean(equals.type(torch.FloatTensor))
                train_acc += acc.item()

                loss = self.criterion(out.squeeze(), target.float())
                train_loss += loss.item()
                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

                self.optimizer.step()

                del feature, target, predicted

            print("Avg loss", train_loss / len(train_loader))
            print("Accuracy", train_acc / len(train_loader))
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['train_acc'].append(train_acc / len(train_loader))

    def test_model(self, test_loader):
        self.eval()  # Set the model to evaluation mode
        test_loss = 0
        test_acc = 0

        all_target = []
        all_predicted = []

        testloop = tqdm(test_loader, leave=True, desc='Inference')

        with torch.no_grad():
            for feature, target in testloop:

                out = self.forward(feature)

                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5])
                equals = predicted == target
                acc = torch.mean(equals.type(torch.FloatTensor))
                test_acc += acc.item()

                loss = self.criterion(out.squeeze(), target.float())
                test_loss += loss.item()

                all_target.extend(target.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
            print(f'Accuracy: {test_acc / len(test_loader):.4f}, Loss: {test_loss / len(test_loader):.4f}')
        print(classification_report(all_predicted, all_target))

        return { 'pred_y': all_predicted, 'true_y': all_target}

    def run(self):

        print("size of train:", self.data['train']['X'].shape, self.data['train']['y'].shape)
        print("size of test:", self.data['test']['X'].shape, self.data['test']['y'].shape)

        print("size of each entry", self.data['train']['X'][0].shape)
        print("check", self.data['train']['X'][0], self.data['train']['y'][0])

        self.initialize_embeddings(self.data['embedding'])
        print("initialized embedding to glove embedding")

        train_dataset = TensorDataset(self.data['train']['X'], self.data['train']['y'])
        test_dataset = TensorDataset(self.data['test']['X'], self.data['test']['y'])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # for epoch in range(self.max_epochs):
        #     print(f'Epoch {epoch + 1}/{self.max_epochs}')
        self.train_model(train_loader)
        test_results = self.test_model(test_loader)

        # print("test results", test_results)

        self.plot_loss()
        self.plot_accuracy()


        # print("pred_y size", len(np.array(test_results['pred_y'])))
        # print("true_y size", len(test_results['true_y']))

        return {'pred_y': test_results['pred_y'], 'true_y': test_results['true_y']}

    def plot_loss(self):
        plt.plot(self.history['train_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig('../../result/stage_4_result/classification_loss.png')
        plt.clf()

    def plot_accuracy(self):
        plt.plot(self.history['train_acc'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.savefig('../../result/stage_4_result/classification_accuracy.png')
        plt.clf()
