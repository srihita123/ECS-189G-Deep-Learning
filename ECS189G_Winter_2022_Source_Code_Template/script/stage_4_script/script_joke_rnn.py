from src_code.stage_4_code.Joke_Dataset_Loader import Joke_Dataset_Loader
from src_code.stage_4_code.Method_Joke_RNN import Method_Joke_RNN
from src_code.stage_4_code.Result_Saver import Result_Saver
from src_code.stage_4_code.Setting_Separate_Train_Test import Setting_Separate_Train_Test
from src_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

#---- Joke RNN text generation script---------------------
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    # dataset name and description
    data_obj = Joke_Dataset_Loader('training data', '')
    data_obj.dataset_source_file_name = 'data'
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'

    # move data loading to initialization step
    method_obj = Method_Joke_RNN('text generation RNN', '', data_obj.load())

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/Joke_RNN_'
    result_obj.result_destination_file_name = 'result'

    setting_obj = Setting_Separate_Train_Test('setting text generation RNN', '')

    evaluate_obj = Evaluate_Metrics('multiple metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    eval_metrics = setting_obj.load_run_save_evaluate()
    print('************ Performance ************')
    print(str(eval_metrics))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    