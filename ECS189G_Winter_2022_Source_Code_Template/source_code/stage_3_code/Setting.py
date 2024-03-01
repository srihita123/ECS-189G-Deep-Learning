# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from source_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting(setting):

    def load_run_save_evaluate(self):
        # Load dataset
        loaded_data = self.dataset.load()

        # Access correct keys in the loaded data dictionary
        X_train = loaded_data['X_train']
        y_train = loaded_data['y_train']
        X_test = loaded_data['X_test']
        y_test = loaded_data['y_test']

        # Run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # Save raw ResultModule

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()
