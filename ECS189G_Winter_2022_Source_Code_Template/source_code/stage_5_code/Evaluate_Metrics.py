'''
Concrete Evaluate class for multiple evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from source_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class Evaluate_Metrics(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        # Return a string with multiple evaluation metrics listed
        return ('Accuracy: ' + str(accuracy_score(self.data['true_y'], self.data['pred_y'])) + '\n'
                + 'Precision: ' + str(precision_score(self.data['true_y'], self.data['pred_y'], average='macro')) + '\n'
                + 'Recall: ' + str(recall_score(self.data['true_y'], self.data['pred_y'], average='macro')) + '\n'
                + 'F1: ' + str(f1_score(self.data['true_y'], self.data['pred_y'], average='macro')))
