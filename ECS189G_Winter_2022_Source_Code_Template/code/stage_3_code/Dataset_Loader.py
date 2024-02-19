# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import os

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()

        for instance in data['train']:
            X_train.append(instance['image'])
            y_train.append(instance['label'])
        for instance in data['test']:
            X_test.append(instance['image'])
            y_test.append(instance['label'])


        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}





