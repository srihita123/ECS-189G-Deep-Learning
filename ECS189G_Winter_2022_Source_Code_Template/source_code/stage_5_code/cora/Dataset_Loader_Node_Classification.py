'''
Concrete IO class for a specific dataset
'''
import random

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from source_code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp

class Dataset_Loader(dataset):
    data = None
    dataset_name = None
    dataset_source_folder_path = None
    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        #return torch.sparse.FloatTensor(indices, values, shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt(self.dataset_source_folder_path + self.dataset_source_file_name + "/node",
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        print("initial features", features[0][0])
        print("initial labels", onehot_labels[0])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(self.dataset_source_folder_path + self.dataset_source_file_name + "/link",
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        # Construct an adjacency matrix from edges (dimensions |V|x|V|)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        print("filtering", features.shape, labels.shape, adj.shape)

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        if self.dataset_name == 'cora':
            # idx_train = range(140)
            # idx_test = range(200, 1250)
            # idx_val = range(1300, 1500)
            unique_labels = torch.unique(labels)
            idx_train = []
            idx_test = []
            idx_val = []

            for label in unique_labels:
                all_idx_label = (labels == label).nonzero().view(-1)
                all_idx_label = all_idx_label[torch.randperm(all_idx_label.size(0))]

                print("all idx", label, all_idx_label)

                train_labels = all_idx_label[:20]
                test_labels = all_idx_label[21: 21 + 150]
                val_labels = all_idx_label[21+150+1: 21+150+1+200]

                print("train size", train_labels.tolist(), len(train_labels))
                print("test size", len(test_labels))

                for idx in train_labels.tolist():
                    idx_train.append(idx)

                for idx in test_labels.tolist():
                    idx_test.append(idx)

                for idx in val_labels.tolist():
                    idx_val.append(idx)

            print("after separating", len(idx_train), len(idx_test), len(idx_val))
            # print(torch.cat(idx_train))
            random.shuffle(idx_train)
            random.shuffle(idx_test)
            random.shuffle(idx_val)

        elif self.dataset_name == 'citeseer':
            idx_train = range(120)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'pubmed':
            idx_train = range(60)
            idx_test = range(6300, 7300)
            idx_val = range(6000, 6300)
        #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        else:
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        # get the training nodes/testing nodes
        # train_x = features[idx_train]
        # val_x = features[idx_val]
        # test_x = features[idx_test]
        # print(train_x, val_x, test_x)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}

