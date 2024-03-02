from source_code.stage_4_code.Dataset_Loader import Dataset_Loader
from source_code.stage_4_code.Result_Saver import Result_Saver
from source_code.stage_4_code.Setting import Setting
from source_code.stage_4_code.Method_Classification import Method_Classification
from source_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch
import json

#---- CNN script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- object initialization section ------------------
    # combine train and test cleaned data
    path_to_train = "clean_reviews_train.json"
    path_to_test = "clean_reviews_train.json"
    # Step 1: Read the first JSON file
    with open(path_to_train, 'r') as file:
        data1 = json.load(file)
    data1['pos'] = data1['pos']
    data1['neg'] = data1['neg']
    # Step 2: Read the second JSON file
    with open(path_to_test, 'r') as file:
        data2 = json.load(file)
    data2['pos'] = data2['pos']
    data2['neg'] = data2['neg']

    combined_data = {
        'train': data1,
        'test': data2
    }
    # print("combined data",combined_data)
    output_file_path = "cleaned_data.json"

    with open(output_file_path, 'w') as file:
        json.dump(combined_data, file, indent=4)

    print("file created")

    data_obj = Dataset_Loader('Classification', '')
    data_obj.dataset_source_folder_path = './'
    data_obj.dataset_source_file_name = 'cleaned_data.json'

    method_obj = Method_Classification('text_classification', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/Text_Classification_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting('setting from file with training and testing', '')

    evaluate_obj = Evaluate_Metrics('multiple metrics', '')

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    result = setting_obj.load_run_save_evaluate()
    print(result)
    print('************ Finish ************')
    # ------------------------------------------------------

