'''
    Concrete class for experiment with separate testing and training data
    Does not implement setting base class!
'''
from code.base_class.setting import setting


class Setting(setting):

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        X_train, X_test, y_train, y_test = loaded_data['X_train'], loaded_data['X_test'], loaded_data['y_train'], \
            loaded_data['y_test']

        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()
