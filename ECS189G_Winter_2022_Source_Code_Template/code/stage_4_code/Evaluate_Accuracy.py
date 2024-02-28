'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        total = 0
        length = len(self.data['true_y'])
        for i in range(length):
            total += accuracy_score(self.data['true_y'][i], self.data['pred_y'][i])
        return total / length
        