'''from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from source_code.stage_5_code.utils import load_data, accuracy, precision, recall, f1_score
from source_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from source_code.stage_5_code.models import GCN

seed = 16
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

curr = Dataset_Loader()
curr.dataset_source_folder_path = 'C:\\Users\\sindura\\Downloads\\ECS189G_Winter_2022_Source_Code_Template\\ECS189G_Winter_2022_Source_Code_Template\\data\\stage_5_data\\citeseer'
curr.dataset_name = "citeseer"

loaded_data = curr.load()


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=16, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout rate (1 - keep probability).')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = loaded_data

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item()

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    prec_test = precision(output[idx_test], labels[idx_test])
    rec_test = recall(output[idx_test], labels[idx_test])
    f1_test = f1_score(output[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "precision = {:.4f}".format(prec_test),
          "recall = {:.4f}".format(rec_test),
          "f1_score = {:.4f}".format(f1_test))

    return loss_test.item(), acc_test.item(), prec_test, rec_test, f1_test

# Train model
t_total = time.time()
train_losses = []
test_losses = []
for epoch in range(args.epochs):
    train_loss, _, _, _ = train(epoch)
    test_loss, _, _, _, _ = test()
    train_losses.append(train_loss)
    test_losses.append(test_loss)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Plot loss over epochs for test
plt.plot(train_losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()'''

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from source_code.stage_5_code.utils import accuracy, precision, recall, f1_score
from source_code.stage_5_code.models import GCN
import matplotlib.pyplot as plt
def train(epoch, adj, features, labels, idx_train, idx_val, idx_test, model, optimizer,args,loss):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    loss.append(loss_val.item())
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test(adj, features, labels, idx_train, idx_val, idx_test, model):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    precision_test = precision(output[idx_test], labels[idx_test])
    recall_test = recall(output[idx_test], labels[idx_test])
    f1_test = f1_score(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "precision= {:.4f}".format(precision_test),
          "recall = {:.4f}".format(recall_test),
          "f1 score = {:.4f}".format(f1_test))

def plot_learning_curves(train_losses, val_losses):
    plt.plot([i for i in range(train_losses)], val_losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.show()