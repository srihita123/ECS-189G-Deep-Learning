import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src_code.stage_4_code.Dataset_Loader import Dataset_Loader

curr = Dataset_Loader()
curr.dataset_source_folder_path = "/Users/Srihita/Desktop/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_generation/"
curr.dataset_source_file_name = "data"
input, output = curr.load()
jokes = curr.jokesProcess()
vocab = curr.vocabSet(jokes)


def testing(input):
    # input : list of 3 words
    encoded_input = curr.encode(input, vocab)
    encoded_input = encoded_input[0] + [0 for _ in range(37)]
    encoded_input = torch.tensor(encoded_input)
    output, _ = model(torch.unsqueeze(encoded_input,1), torch.zeros(1,40))
    output = torch.flatten(output)
    output = output.tolist()
    decoded_output = curr.decodeTest(output, vocab)
    return decoded_output
# Define RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        #self.embedding = nn.Embedding(len(vocab), hidden_size)
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        #embedded = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


# Hyperparameters
input_size = len(input) + 1  # Vocabulary size
hidden_size = 40
output_size = len(output) + 1  # Vocabulary size
print("Output size:",output)
print("Output row:",len(output[1]))

learning_rate = 0.001
num_epochs = 100
batch_size = 32

# Convert data to PyTorch tensors
input_tensors = torch.FloatTensor(input)
output_tensors = torch.FloatTensor(output)

# Create DataLoader
dataset = TensorDataset(input_tensors, output_tensors)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for input_seq, target_seq in dataloader:
        optimizer.zero_grad()
        hidden = model.init_hidden(input_seq.size(0))
        output, _ = model(torch.unsqueeze(input_seq,2), hidden)
        #padding_sizes = (0, 0, 3)
        #padding_tensor = torch.zeros(output.size(0), output.size(1), padding_sizes[2],dtype=output.dtype)
        #padded_output_tensor = torch.cat((output, padding_tensor), dim=2)
        #print("Padded Output Tensor Shape:", padded_output_tensor.shape)

        loss = criterion(output, target_seq.unsqueeze(2))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}')


print(testing([["what", "did", "the"]]))
