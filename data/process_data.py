import numpy as np
import pandas as pd


class ProcessData():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols, target_col):
        df = pd.read_csv(filename)
        self.train_split = int(len(df) * split)
        self.dataset = df.get(cols).values
        self.target_col = target_col
        self.target = self.dataset[:, self.target_col]

        self.train_data_mean = df.get(cols).values[:self.train_split].mean(axis=0)
        self.train_data_std = df.get(cols).values[:self.train_split].std(axis=0)
        self.test_data_mean = df.get(cols).values[self.train_split:].mean(axis=0)
        self.test_data_std = df.get(cols).values[self.train_split:].std(axis=0)

    def get_labeled_data(self, history_length, target_length, step, train_data=True, single_step=False):
        feature_list = []
        label_list = []
        if train_data:
            start_index = 0 + history_length
            end_index = self.train_split
        else:
            start_index = self.train_split
            end_index = len(self.dataset) - target_length

        for i in range(start_index, end_index):
            indices = range(i - history_length, i, step)
            feature_list.append(self.dataset[indices])

            if single_step:
                label_list.append(self.target[i + target_length])
            else:
                label_list.append(self.target[i:i + target_length])

        return np.array(feature_list), np.array(label_list)

    def normalize_data(self):
        train_set = (self.dataset[:self.train_split] - self.train_data_mean) / self.train_data_std
        test_set = (self.dataset[self.train_split:] - self.test_data_mean) / self.test_data_std
        train_target = (self.target[:self.train_split] - self.train_data_mean[self.target_col]) \
                       / self.train_data_std[self.target_col]
        test_target = (self.target[self.train_split:] - self.test_data_mean[self.target_col]) \
                      / self.test_data_std[self.target_col]
        self.dataset = np.concatenate((train_set, test_set))
        self.target = np.concatenate((train_target, test_target))

    def denormalize_test_data(self, y):
        return y * self.test_data_std[self.target_col] + self.test_data_mean[self.target_col]
