import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from code.stage_4_code.Dataset_Loader import Dataset_Loader

curr = Dataset_Loader()
curr.dataset_source_folder_path = "../../data/stage_4_data/text_generation/"
curr.dataset_source_file_name = "data"
input, output = curr.load()
jokes = curr.jokesProcess()
vocab_to_int, int_to_vocab = curr.vocabSet(jokes)

def rand_choice(prob):
    prob /= np.sum(prob)
    index = np.random.choice(len(int_to_vocab), p=prob)
    return int_to_vocab[index]

def testing(input):
    # input : list of 3 words
    encoded_input = curr.encode(input, vocab_to_int)
    encoded_input = encoded_input[0] + [0 for _ in range(37)]
    encoded_input = torch.tensor(encoded_input)
    output, _ = model(torch.unsqueeze(encoded_input,0), model.init_hidden(1))
    output = torch.detach(output)
    output_list = torch.chunk(output, chunks=output.size(1), dim=1)
    decoded_output = []
    for prob in output_list:
        decoded_output.append(rand_choice(prob.numpy().reshape(-1)))
    return decoded_output
# Define RNN model
class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)

        # self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        # output, hidden = self.rnn(x, hidden)
        out, hidden = self.lstm(embedded, hidden)
        out = self.softmax(self.fc(out))
        return out, hidden

    def init_hidden(self, batch_size):
        # return h_0 and c_0
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


# Hyperparameters
hidden_size = 200
num_layers = 2
vocab_size = len(vocab_to_int)

learning_rate = 0.001
num_epochs = 30
batch_size = 400
embedding_size = 200

# Convert data to PyTorch tensors
input_tensors = torch.LongTensor(input)
output_tensors = torch.LongTensor(output)

# Create DataLoader
dataset = TensorDataset(input_tensors, output_tensors)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = RNN(vocab_size, hidden_size, embedding_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for input_seq, target_seq in dataloader:
        # optimizer.zero_grad()
        # hidden = model.init_hidden(input_seq.size(0))
        # output, _ = model(torch.unsqueeze(input_seq,2), hidden)
        # loss = criterion(output, target_seq.unsqueeze(2))
        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item()

        optimizer.zero_grad()
        hidden = model.init_hidden(input_seq.size(0))
        model_out, _ = model(input_seq, hidden)

        # Flatten the target tensor to make it compatible with CrossEntropyLoss
        target_flat = target_seq.view(-1)
        # Reshape the model output tensor to [batch_size * sequence_length, vocab_size]
        model_out_flat = model_out.view(-1, model_out.size(-1))
        # Calculate the CrossEntropyLoss
        loss = criterion(model_out_flat, target_flat)

        #loss = criterion(model_out, target_seq.unsqueeze(2))
        loss.backward()
        # Gradient clipping to prevent exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        total_loss += loss.item()

        # what
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}')


print(testing([["what", "did", "the"]]))
