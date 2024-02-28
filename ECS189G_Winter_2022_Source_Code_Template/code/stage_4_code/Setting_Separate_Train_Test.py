'''
    Concrete class for experiment with separate testing and training data
    Does not implement setting base class!
'''
from src_code.base_class.setting import setting
class Setting_Separate_Train_Test(setting):

    # Add new attribute for testing dataset
    testset = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print('dataset:', self.dataset.dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name,
              ', evaluation:', self.evaluate.evaluate_name)

    def load_run_save_evaluate(self):
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()
