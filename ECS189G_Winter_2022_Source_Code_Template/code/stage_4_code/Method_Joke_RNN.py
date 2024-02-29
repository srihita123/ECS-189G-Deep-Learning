'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Joke_Dataset_Loader import Joke_Dataset_Loader
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


# Define the RNN model
class Method_Joke_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 5
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 2e-3
    hidden_size = 500
    training = None

    # NOTE: changed implementation so you need to provide dataset at initialization
    def __init__(self, mName, mDescription, data, training=True):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.data = data
        encoding_size = len(self.data['vocab'])
        # each rnn block takes a one-hot encoding of token and outputs result of size hidden_size
        self.rnn = nn.RNN(input_size=encoding_size, hidden_size=self.hidden_size, nonlinearity='relu')
        # output layer takes input from final rnn state and outputs vector of probabilities
        self.out = nn.Linear(self.hidden_size, encoding_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.training = training

    def forward(self, x):
        '''Forward propagation'''
        output, hidden = self.rnn(x)
        # pass only the final hidden state to output layer
        output = self.softmax(self.out(hidden))
        return output.squeeze(0)

    def train(self):
        # skip over training if flag set to false
        if not self.training:
            self.load_state_dict(torch.load('joke_rnn_weights.pth'))
            print('Loaded previous training weights.')
            return
        # List to store loss values during training
        losses = []
        # set optimizer, loss function, and evaluation criteria
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # self.data['doc'] contains a list of jokes, which are variable length lists of numpy arrays.
        # Each numpy array in joke is an encoding of a token
        # Input to model should have shape (max joke length x number of jokes x token encoding size)
        joke_tensor = nn.utils.rnn.pad_sequence([torch.FloatTensor(np.array(joke)) for joke in self.data['doc']])
        # Shorter jokes have zero padding at the end
        # end_of_joke = encoding = np.zeros(len(self.vocab))
        max_joke_len = joke_tensor.size(0)

        for epoch in range(self.max_epoch):
            total_loss = 0
            optimizer.zero_grad()
            # List of all token predictions indices [3, max_joke_len - 1]
            pred_y = []
            for i in range(max_joke_len - 3):  # Loop over token indices
                token_window = joke_tensor[i:i+3, :, :]
                next_token_pred = self.forward(token_window)
                total_loss += loss_function(next_token_pred, joke_tensor[i+3, :, :])
                pred_y.append(next_token_pred.max(1)[1])

            losses.append(total_loss.item())
            total_loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                # trying to get the true words for indices 3 thru end, turn them into a list
                # joke_tensor[3:max_joke_len, :, :].max(2)[1] is 37 x 1622
                y_true = list(torch.tensor_split(joke_tensor[3:max_joke_len, :, :].max(2)[1], max_joke_len - 3, dim=0))
                y_true = [i.squeeze(0) for i in y_true]
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': pred_y}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', total_loss.item())

        torch.save(self.state_dict(), 'joke_rnn_weights.pth')
        # Save training convergence plot
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig('../../result/stage_4_result/joke_rnn_loss.png')
        return

    # Test function to generate joke from trained model
    # Expects a list of three strings
    def test(self, test_data, max_joke_len=45):
        with torch.no_grad():
            encoded_test_data = test_data.get_encoded_single()
            encoded_test_data = [np.array(encoding) for encoding in encoded_test_data]
            end_of_joke_idx = test_data.vocab.index("END_OF_JOKE")

            for i in range(max_joke_len - 3):  # Loop over token indices
                token_window = torch.FloatTensor(encoded_test_data[i:i+3])
                # get the index of next token that is most likely
                next_token_idx = self.forward(token_window).max(0)[1].item()
                if next_token_idx == end_of_joke_idx:
                    break
                next_token_encoding = torch.FloatTensor(test_data.encode(test_data.vocab[next_token_idx]))
                encoded_test_data = encoded_test_data.append(next_token_encoding)
        return encoded_test_data

    def run(self):
        print('method running...')
        print('--start training...')
        self.train()

        print('--start testing...')
        test_data = Joke_Dataset_Loader()
        test_data.data = ['Why', 'did', 'the']
        test_data.vocab = self.data['vocab']
        joke_prediction = self.test(test_data)
        print(joke_prediction)
        return 1
