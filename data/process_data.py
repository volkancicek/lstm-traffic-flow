import numpy as np
import pandas as pd
import os


class ProcessData:
    """A class for loading and transforming data for RNN models"""

    def __init__(self, configs):
        self.configs = configs
        self.columns = configs['data']['columns']
        self.df = pd.read_csv(os.path.join('data', configs['data']['approach_1']['data_file_name']))
        self.train_split = int(len(self.df) * configs['data']['train_test_split'])
        self.target_col = configs['data']['target_column']
        self.feature_columns = configs['data']['feature_columns']
        self.features = self.df.get(self.columns).values[:, self.feature_columns]
        self.target = self.df.get(self.columns).values[:, self.target_col]
        self.train_data_mean = self.df.get(self.columns).values[:self.train_split].mean(axis=0)
        self.train_data_std = self.df.get(self.columns).values[:self.train_split].std(axis=0)
        self.test_data_mean = self.df.get(self.columns).values[self.train_split:].mean(axis=0)
        self.test_data_std = self.df.get(self.columns).values[self.train_split:].std(axis=0)

    def get_labeled_data(self, train_data=True, single_step=True):
        history_length = self.configs['data']['history_size']
        target_length = self.configs['data']['target_range']
        step = self.configs['data']['step']
        feature_list = []
        label_list = []
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
            else:
                label_list.append(self.target[i:i + target_length])

        return np.array(feature_list), np.array(label_list)

    def normalize_data(self):
        train_set = (self.df.get(self.columns).values[:self.train_split] - self.train_data_mean) / self.train_data_std
        test_set = (self.df.get(self.columns).values[self.train_split:] - self.test_data_mean) / self.test_data_std
        dataset = np.concatenate((train_set, test_set))
        self.target = dataset[:, self.target_col]
        self.features = dataset[:, self.feature_columns]

    def denormalize_target(self, y):
        return y * self.test_data_std[self.target_col] + self.test_data_mean[self.target_col]
