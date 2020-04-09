import numpy as np
import pandas as pd
import matplotlib as plt


class ProcessData():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols, target_col):
        df = pd.read_csv(filename)
        self.train_split = int(len(df) * split)
        self.dataset = df.get(cols).values
        self.target_col = target_col
        self.target = self.dataset[:, self.target_col]
        """
        self.data_train = df.get(cols).values[:i_split]
        self.data_test = df.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None
        """
        self.data_mean = self.dataset.mean(axis=0)
        self.data_std = self.dataset.std(axis=0)

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

    def normalise_data(self):
        self.dataset = (self.dataset - self.data_mean) / self.data_std
        self.target = (self.target - self.data_mean[self.target_col]) / self.data_std[self.target_col]

