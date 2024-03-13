from source_code.base_class.setting import setting


class Cora_Setting(setting):

    # Get adjacency matrix for training / testing subset of nodes
    def filter_adj(self, adj_matrix, indices):
        dense = adj_matrix.to_dense()
        filtered_adj = dense[indices][:, indices].to_sparse()
        return filtered_adj

    def load_run_save_evaluate(self):
        # load training dataset and testing dataset separately
        # DIFFERENT Setting_Train_Test_Split, which splits from single dataset
        data = self.dataset.load()
        adj = data['graph']['utility']['A']

        # Split into training and testing
        X_train = data['graph']['X'][data['train_test_val']['idx_train']]
        X_test = data['graph']['X'][data['train_test_val']['idx_test']]
        y_train = data['graph']['y'][data['train_test_val']['idx_train']]
        y_test = data['graph']['y'][data['train_test_val']['idx_test']]

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        # Filter adjacency list to only include nodes in set
        adj_train = self.filter_adj(adj, data['train_test_val']['idx_train'])
        adj_test = self.filter_adj(adj, data['train_test_val']['idx_test'])

        self.method.data = {'train': {'X': X_train, 'y': y_train, 'adj': adj_train},
                            'test': {'X': X_test, 'y': y_test, 'adj': adj_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()