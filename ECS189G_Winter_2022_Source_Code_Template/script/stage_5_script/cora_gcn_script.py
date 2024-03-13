from source_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from source_code.stage_5_code.Cora_GCN_Method2 import Cora_GCN_Method
from source_code.stage_5_code.Result_Saver import Result_Saver
from source_code.stage_5_code.Cora_Setting import Cora_Setting
from source_code.stage_5_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    dataset_obj = Dataset_Loader('Cora dataset', '')
    dataset_obj.dataset_source_folder_path = "../../data/stage_5_data/"
    dataset_obj.dataset_source_file_name = "cora"
    dataset_obj.dataset_name = "cora"

    method_obj = Cora_GCN_Method('GCN on Cora dataset', '', nfeat=1433, nclass=7)

    result_obj = Result_Saver('Cora saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/Cora'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Cora_Setting('Setting Cora', '')

    evaluate_obj = Evaluate_Metrics('multiple metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(dataset_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    eval_metrics = setting_obj.load_run_save_evaluate()
    print('************ Performance ************')
    print(str(eval_metrics))
    print('************ Finish ************')
    # ------------------------------------------------------


