from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting import Setting
from code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    # dataset name and description
    data_obj = Dataset_Loader('CIFAR', '')
    data_obj.dataset_source_file_name = 'CIFAR'
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'

    use_mps = False
    method_obj = Method_CNN_CIFAR('cnn_cifar_dataset', '', mps=use_mps)
    # Use MPS if available
    if use_mps and torch.backends.mps.is_available():
        method_obj.to(torch.device("mps"))

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting('setting from file with training and testing', '')

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
