from code.stage_4_code.JokeDataLoader import JokeDataLoader
from code.stage_4_code.JokeRNNMethod import JokeRNNMethod
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting import Setting
from code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

# ---- Joke RNN script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    # dataset name and description
    data_obj = JokeDataLoader('Joke data', '')
    data_obj.dataset_source_file_name = 'data'
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'

    method_obj = JokeRNNMethod('Joke RNN', '', data_obj)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/'
    result_obj.result_destination_file_name = 'generated_jokes.txt'

    setting_obj = Setting('setting for Joke generation', '')

    evaluate_obj = Evaluate_Metrics('multiple metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    generated_text = setting_obj.load_run_save_evaluate("How do dogs", training=False)
    print('************ Generated Text ************')
    print(generated_text)
    print('************ Finish ************')
    # ------------------------------------------------------


