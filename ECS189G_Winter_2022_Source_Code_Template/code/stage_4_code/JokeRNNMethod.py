import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from code.stage_4_code.JokeDataset import JokeDataset
from code.base_class.method import method

# Define RNN model
class JokeRNNMethod(nn.Module, method):
    data = None
    dataloader = None
    max_epochs = 20
    learning_rate = 1e-3
    dropout = 0.2
    # affects diversity of output probabilities before passing to softmax
    temperature = 1.5

    rnn_size = 500
    embedding_size = 200
    num_layers = 3
    batch_size = 1000
    # Given 3 words, predict the next one
    seq_len = 3

    def __init__(self, mName, mDescription, dataloader):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # Load training data at model initialization
        dataloader.load()
        self.dataloader = dataloader
        self.embedding = nn.Embedding(self.dataloader.vocab_size, self.embedding_size)
        self.rnn = nn.LSTM(self.embedding_size, self.rnn_size, self.num_layers, dropout=self.dropout, batch_first=True)

        # self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(self.rnn_size, self.dataloader.vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        # output, hidden = self.rnn(x, hidden)
        out, hidden = self.rnn(embedded)
        # Only feed the output from last time step to FC layer
        final_out = self.fc(out[:, -1, :])
        return final_out, hidden

    def init_hidden(self, batch_size):
        # return h_0 and c_0
        return (torch.zeros(self.num_layers, batch_size, self.rnn_size),
                torch.zeros(self.num_layers, batch_size, self.rnn_size))

    def train_custom(self):
        losses = []
        # Create Pytorch dataset from loaded data
        self.data = JokeDataset(self.dataloader.encoded_data, self.seq_len)
        # Create a Pytorch dataloader for mini-batching
        batch_dataloader = DataLoader(self.data, batch_size=self.batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # For keeping track of best weights
        best_model = None
        best_loss = np.inf
        # Training loop
        for epoch in range(self.max_epochs):
            total_loss = 0
            for batch, (X, y) in enumerate(batch_dataloader):
                optimizer.zero_grad()
                y_pred, hidden = self.forward(X)
                loss = criterion(y_pred, y)
                loss.backward()
                # Gradient clipping to prevent exploding gradient
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
                optimizer.step()
                print("Batch: ", batch)
            self.eval()
            for batch, (X, y) in enumerate(batch_dataloader):
                # Save best weights
                y_pred, hidden = self.forward(X)
                loss = criterion(y_pred, y)
                total_loss += loss.item()
                if loss < best_loss:
                    best_loss = loss
                    best_model = self.state_dict()
            losses.append(total_loss)
            print(f'Epoch {epoch+1}/{self.max_epochs}, Avg Loss: {total_loss / len(batch_dataloader):.4f}')
        # Save best weights from training
        torch.save([best_model, self.dataloader.vocab_to_int], "../../code/stage_4_code/joke-weights.pth")

        # Save training convergence plot
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.savefig('../../result/stage_4_result/joke_loss.png')

    def test(self, start_tokens, max_tokens=10):
        # Load weights from best model found during training
        best_model, vocab_to_int = torch.load("../../code/stage_4_code/joke-weights.pth")
        self.load_state_dict(best_model)
        # Encode start token input
        encoding = self.dataloader.encode_text(start_tokens)
        seq_len = len(encoding)
        for i in range(max_tokens):
            if vocab_to_int["END_OF_JOKE"] in encoding[i: i + seq_len]:
                break  # Stop generation if "END_OF_JOKE" token is in the window
            # Note: model expects batches, so create batch of 1
            X = torch.tensor([encoding[i: i+seq_len]])
            y_pred, hidden = self.forward(X)
            # Note: model returns batched output, so remove singleton batch dimension
            y_pred = y_pred.squeeze(0)
            # Pick next token based on output distribution
            probabilities = torch.nn.functional.softmax(y_pred, dim=0)
            index = torch.multinomial(probabilities, 1).item()
            # Add word to end of encoding
            encoding.append(index)
        # Remove end of joke token if last generated token
        if encoding[-1] == vocab_to_int["END_OF_JOKE"]:
            encoding.pop()

        decoded = self.dataloader.decode_text(encoding)
        return ' '.join(decoded)

    def run(self, start_tokens, training=False):
        print('method running...')
        if training:
            print('--start training...')
            self.train_custom()
        print('--start testing...')
        pred_y = self.test(start_tokens)
        return pred_y

'''
dataloader = JokeDataLoader()
dataloader.dataset_source_folder_path = "../../data/stage_4_data/text_generation/"
dataloader.dataset_source_file_name = "data"

model = JokeRNNMethod("", "", dataloader)
print(model.run())
'''