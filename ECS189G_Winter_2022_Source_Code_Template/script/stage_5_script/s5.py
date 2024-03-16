from __future__ import print_function
import time
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from source_code.stage_5_code.utils import load_data
from source_code.stage_5_code.models import GCN
from source_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from source_code.stage_5_code.train import train, plot_learning_curves, test

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

curr = Dataset_Loader()
curr.dataset_name = 'pubmed'
curr.dataset_source_folder_path = "/Users/Srihita/Desktop/ECS189G_Winter_2022_Source_Code_Template/data/stage_5_data/pubmed/"

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
loss = []
epochs = args.epochs
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = curr.load()

# Model and optimizer
model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch, adj, features, labels, idx_train, idx_val, idx_test, model, optimizer,args,loss)
plot_learning_curves(70, loss)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test(adj, features, labels, idx_train, idx_val, idx_test, model)
