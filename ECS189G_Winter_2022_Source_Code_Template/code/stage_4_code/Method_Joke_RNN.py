'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src_code.base_class.method import method
from src_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


# Define the RNN model
class Method_Joke_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 2e-3
    hidden_size = 500

    # NOTE: changed implementation so you need to provide dataset at initialization
    def __init__(self, mName, mDescription, data):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.data = data

        encoding_size = len(self.data['vocab'])
        # each rnn block takes a one-hot encoding of token and outputs result of size hidden_size
        self.rnn = nn.RNN(input_size=encoding_size, hidden_size=self.hidden_size)
        # output layer takes input from final rnn state and outputs vector of probabilities
        self.out = nn.Linear(self.hidden_size, encoding_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        '''Forward propagation'''
        output, hidden = self.rnn(x)
        # pass only the final hidden state to output layer
        output = self.softmax(self.out(hidden))
        return output.squeeze(0)

    def train(self):
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

    # Test function to generate joke from trained model
    '''
    def test(self, start_tokens, max_length=100):
        with torch.no_grad():
            input = start_tokens
            hidden = torch.zeros(1, 1, self.hidden_size)

            joke = start_tokens
            for _ in range(max_length):
                output, hidden = model(input, hidden)
                _, topi = output.topk(1)
                next_token = topi.squeeze().item()
                if next_token == end_of_joke_token:
                    break
                joke.append(next_token)
                input = torch.tensor([[next_token]])
        return joke
    '''

    def run(self):
        print('method running...')
        print('--start training...')
        self.train()
        print('--start testing...')
        return 1
