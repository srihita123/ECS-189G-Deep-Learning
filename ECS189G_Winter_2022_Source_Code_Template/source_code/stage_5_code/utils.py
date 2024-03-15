import numpy as np
import scipy.sparse as sp
import torch
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def precision(output, labels):
    preds = output.argmax(dim=1)  # Predicted class indices
    true_positives = ((preds == 1) & (labels == 1)).sum().item()  # Count true positives
    false_positives = ((preds == 1) & (labels == 0)).sum().item()  # Count false positives

    # Avoid division by zero if no positive predictions are made
    if true_positives + false_positives == 0:
        return 0.0

    # Calculate precision
    precision = true_positives / (true_positives + false_positives)
    return precision


def recall(output, labels):
    preds = output.argmax(dim=1)  # Predicted class indices
    true_positives = ((preds == 1) & (labels == 1)).sum().item()  # Count true positives
    false_negatives = ((preds == 0) & (labels == 1)).sum().item()  # Count false negatives

    # Avoid division by zero if no actual positive cases exist
    if true_positives + false_negatives == 0:
        return 0.0

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives)
    return recall


def f1_score(output, labels):
    # Calculate precision and recall
    precision_val = precision(output, labels)
    recall_val = recall(output, labels)

    # Avoid division by zero if precision or recall is 0
    if precision_val + recall_val == 0:
        return 0.0

    # Calculate F1 score using the formula
    f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    return f1_score_val
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
