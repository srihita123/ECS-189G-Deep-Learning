'''
    Concrete class for experiment with separate testing and training data
    Does not implement setting base class!
'''
from code.base_class.setting import setting
class Setting(setting):

    # Add new attribute for testing dataset
    testset = None

    def prepare(self, sData, sMethod, sResult, sEvaluate):
        self.dataset = sData
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print('data:', self.dataset.dataset_name, 'method:',
              self.method.method_name, ', setting:', self.setting_name, ', result:', self.result.result_name,
              ', evaluation:', self.evaluate.evaluate_name)

    def load_run_save_evaluate(self, start_tokens, training):
        # Run method (for JokeRNNMethod, dataloader passed at initialization)
        learned_result = self.method.run(start_tokens, training)
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
        return learned_result
