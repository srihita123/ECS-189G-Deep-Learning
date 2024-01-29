from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Separate_Train_Test import Setting_Separate_Train_Test
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    # dataset name and description
    train_data = Dataset_Loader('train', '')
    train_data.dataset_source_file_name = 'train.csv'
    train_data.dataset_source_folder_path = '../../data/stage_2_data/'

    test_data = Dataset_Loader('test', '')
    test_data.dataset_source_file_name = 'test.csv'
    test_data.dataset_source_folder_path = '../../data/stage_2_data/'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Separate_Train_Test('separate training and testing data', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_data, test_data, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy = setting_obj.load_run_save_evaluate()
    print('************ Performance ************')
    print('MLP Accuracy: ' + str(accuracy))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    