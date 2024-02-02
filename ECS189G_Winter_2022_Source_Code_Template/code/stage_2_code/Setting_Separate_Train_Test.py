'''
    Concrete class for experiment with separate testing and training data
    Does not implement setting base class!
'''
from code.base_class.setting import setting
class Setting_Separate_Train_Test(setting):

    # Add new attribute for testing dataset
    testset = None

    def prepare(self, sTrainset, sTestset, sMethod, sResult, sEvaluate):
        self.dataset = sTrainset
        self.testset = sTestset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print('trainset:', self.dataset.dataset_name, ', testset:', self.testset.dataset_name, 'method:',
              self.method.method_name, ', setting:', self.setting_name, ', result:', self.result.result_name,
              ', evaluation:', self.evaluate.evaluate_name)

    def load_run_save_evaluate(self):
        # load training dataset and testing dataset separately
        # DIFFERENT Setting_Train_Test_Split, which splits from single dataset
        loaded_train = self.dataset.load()
        loaded_test = self.testset.load()
        X_train, X_test, y_train, y_test = loaded_train['X'], loaded_test['X'], loaded_train['y'], loaded_test['y']

        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()