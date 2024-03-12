import torch
import matplotlib.pyplot as plt
import numpy as np
from pygcn import GCN
from source_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from source_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from source_code.stage_5_code.Evaluate_Metrics import Evaluate_Metrics

# Get adjacency matrix for training / testing subset of nodes
def filter_adjacency_matrix(adj_matrix, indices):
    dense = adj_matrix.to_dense()
    filtered_adj = dense[indices][:, indices].to_sparse()
    return filtered_adj

# Fix random seeds
np.random.seed(2)
torch.manual_seed(2)

# Hyperparameters
hidden_size = 300
dropout = 0.1
learning_rate = 1e-3
max_epoch = 20

# Load data
dataloader = Dataset_Loader('Cora dataset', '')
dataloader.dataset_source_folder_path = "../../data/stage_5_data/"
dataloader.dataset_source_file_name = "cora"
dataloader.dataset_name = "cora"
data = dataloader.load()
adj = data['graph']['utility']['A']

# Split into training and testing
X_train = data['graph']['X'][data['train_test_val']['idx_train']]
X_test = data['graph']['X'][data['train_test_val']['idx_test']]
y_train = data['graph']['y'][data['train_test_val']['idx_train']]
y_test = data['graph']['y'][data['train_test_val']['idx_test']]
# Filter adjacency list to only include nodes in set
adj_train = filter_adjacency_matrix(adj, data['train_test_val']['idx_train'])
adj_test = filter_adjacency_matrix(adj, data['train_test_val']['idx_test'])

# Set up model
model = GCN(nfeat=X_train.shape[1], nhid=hidden_size, nclass=y_train.max().item() + 1, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = torch.nn.CrossEntropyLoss()
accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
losses = []

# TRAINING
for epoch in range(max_epoch):  # you can do an early stop if self.max_epoch is too much...
    # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
    y_pred = model(X_train, adj_train)
    # calculate the training loss
    train_loss = loss_function(y_pred, y_train)
    # record loss for training convergence plot
    losses.append(train_loss.item())
    # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
    optimizer.zero_grad()
    # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
    # do the error backpropagation to calculate the gradients
    train_loss.backward()
    # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
    # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
    optimizer.step()

    if epoch % 2 == 0:
        accuracy_evaluator.data = {'true_y': y_train, 'pred_y': y_pred.max(1)[1]}
        print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
# Save training convergence plot
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig('../../result/stage_5_result/cora_loss.png')

# TESTING
model.eval()
y_pred = model(X_test, adj_test).max(1)[1]
print(y_test)
print(y_pred)

# Evaluate testing
evaluate_metrics = Evaluate_Metrics('testing evaluator', '')
evaluate_metrics.data = {'true_y': y_test, 'pred_y': y_pred}
print('Test Metrics\n', evaluate_metrics.evaluate())
