import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from source_code.stage_5_code.source_code_citeseer.utils import accuracy, precision, recall, f1_score
from source_code.stage_5_code.source_code_citeseer.models import GCN
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