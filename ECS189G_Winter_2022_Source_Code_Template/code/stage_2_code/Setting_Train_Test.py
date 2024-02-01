'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

class Setting_Train_Test(setting):
    fold = 3
    test_dataset = None

    def prepare_separate(self, sDatasetTrain, sDatasetTest, sMethod, sResult, sEvaluate):
        self.dataset = sDatasetTrain # train dataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate
        self.test_dataset = sDatasetTest # test dataset

    def graph_loss(self, losses):
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.savefig("../../result/stage_2_result/MLP_loss_graph")
    
    def load_run_save_evaluate(self):
        
        # load train dataset
        loaded_train_data = self.dataset.load()

        # load test dataset
        loaded_test_data = self.test_dataset.load()
        
        # kf = KFold(n_splits=self.fold, shuffle=True)
        
        fold_count = 0
        score_list = {}
        # for _ in range(self.fold):

        fold_count += 1
        print('************ Fold:', fold_count, '************')
        X_train, X_test = np.array(loaded_train_data['X']), np.array(loaded_test_data['X'])
        y_train, y_test = np.array(loaded_train_data['y']), np.array(loaded_test_data['y'])

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.fold_count = fold_count
        self.result.save()

        self.evaluate.data = learned_result
        score_list = self.evaluate.evaluate()

        self.graph_loss(self.method.losses_list)

        return score_list, np.std(score_list['Accuracy'])

        