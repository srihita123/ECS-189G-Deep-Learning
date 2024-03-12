import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from source_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output + self.bias

# Instantiate the Dataset_Loader
curr = Dataset_Loader()
curr.dataset_name = 'pubmed'
curr.dataset_source_folder_path = "../../data/stage_5_data/pubmed"  # Update with the actual path

# Load the dataset
data = curr.load()

# Get dataset information
input_dim = data['graph']['X'].shape[1]  # Input dimension
output_dim = torch.unique(data['graph']['y']).size(0)  # Output dimension
hidden_dim = 120  # Hidden dimension

# Instantiate the GCN model
model = GCN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
def train_model(model, data, epochs):
    model.train()
    features, adj, labels = data['graph']['X'], data['graph']['utility']['A'], data['graph']['y']
    idx_train = data['train_test_val']['idx_train']
    idx_test = data['train_test_val']['idx_test']
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = criterion(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        acc_train = accuracy(output[idx_train], labels[idx_train])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print('Epoch: {:04d} | Loss: {:.4f} | Train Accuracy: {:.4f} | Test Accuracy: {:.4f}'.format(epoch + 1,loss_train.item(),acc_train,acc_test))

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Train the model
train_model(model, data, epochs=50)
