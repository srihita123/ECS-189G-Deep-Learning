from src_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score


class Evaluate_Precision(evaluate):
    data = None

    def evaluate(self):
        print('Precision is: ')
        return precision_score(self.data['true_y'], self.data['pred_y'], average='weighted')




