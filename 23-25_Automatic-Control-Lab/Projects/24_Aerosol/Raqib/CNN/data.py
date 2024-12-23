import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class WindowGenerator:
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def make_windows(self, df):
        data = np.array(df, dtype=np.float32)
        X, Y = [], []
        for i in range(len(data) - self.total_window_size + 1):
            X.append(data[i:i + self.input_width, :-1])  # All except the last column
            Y.append(data[i + self.label_start:i + self.total_window_size, -1])  # Last column

        X = np.array(X)  # Shape: (num_windows, input_width, num_features)
        Y = np.array(Y)  # Shape: (num_windows, label_width)
        return X, Y

    def get_datasets(self, batch_size):
        train_X, train_Y = self.make_windows(self.train_df)
        val_X, val_Y = self.make_windows(self.val_df)
        test_X, test_Y = self.make_windows(self.test_df)

        train_dataset = TensorDataset(torch.tensor(train_X), torch.tensor(train_Y))
        val_dataset = TensorDataset(torch.tensor(val_X), torch.tensor(val_Y))
        test_dataset = TensorDataset(torch.tensor(test_X), torch.tensor(test_Y))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader


def load_data(file):
    df = pd.read_excel(file)
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.8)]
    test_df = df[int(n * 0.8):]
    return train_df, val_df, test_df