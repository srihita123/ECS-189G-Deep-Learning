from source_code.base_class.setting import setting


class Cora_Setting(setting):

    # Get adjacency matrix for training / testing subset of nodes
    def filter_adj(self, adj_matrix, indices):
        dense = adj_matrix.to_dense()
        filtered_adj = dense[indices][:, indices].to_sparse()
        return filtered_adj

    def load_run_save_evaluate(self):
        self.method.data = self.dataset.load()
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()