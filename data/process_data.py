import numpy as np
import pandas as pd
import os


class ProcessData:
    """A class for loading and transforming data for RNN models"""

    def __init__(self, configs):
        self.configs = configs
        df = pd.read_csv(os.path.join('data', configs['data']['approach_1']['data_file_name']))
        self.train_split = int(len(df) * configs['data']['train_test_split'])
        self.target_col = configs['data']['target_column']
        self.feature_columns = configs['data']['feature_columns']
        self.dates = df.get("date").values[:]
        self.features = df.get(self.feature_columns).values[:]
        self.target = df.get(self.target_col).values[:]
        self.train_data_mean = self.features[:self.train_split].mean(axis=0)
        self.train_data_std = self.features[:self.train_split].std(axis=0)
        self.target_mean = self.target[:self.train_split].mean(axis=0)
        self.target_std = self.target[:self.train_split].std(axis=0)



    def get_labeled_data(self, train_data=True, single_step=True):
        history_length = self.configs['data']['history_size']
        target_length = self.configs['data']['target_range']
        step = self.configs['data']['step']
        feature_list = []
        label_list = []
        date_list = []
        if train_data:
            start_index = 0 + history_length
            end_index = self.train_split
        else:
            start_index = self.train_split
            end_index = len(self.features) - target_length

        for i in range(start_index, end_index):
            indices = range(i - history_length, i, step)
            feature_list.append(self.features[indices])

            if single_step:
                label_list.append(self.target[i + target_length])
                date_list.append(self.dates[i + target_length])
            else:
                label_list.append(self.target[i:i + target_length])
                date_list.append(self.dates[i:i + target_length])

        return np.array(feature_list), np.array(label_list), date_list

    def normalize_data(self):
        self.features = (self.features[:] - self.train_data_mean) / self.train_data_std
        self.target = (self.target[:] - self.target_mean) / self.target_std

    def denormalize_target(self, y):
        return (y * self.target_std) + self.target_mean
