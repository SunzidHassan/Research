import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class WindowGenerator:
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)

        # Define input and label columns (all columns except the last one for inputs)
        self.input_columns = train_df.columns[:-1]
        self.label_columns = train_df.columns[-1:]

    def __repr__(self):
        input_indices = list(range(self.input_width))
        label_indices = list(range(self.label_start, self.total_window_size))
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {input_indices}',
            f'Label indices: {label_indices}',
            f'Input column name(s): {list(self.input_columns)}',
            f'Label column name(s): {list(self.label_columns)}'
        ])

    def make_windows(self, df):
        data = np.array(df, dtype=np.float32)
        X, Y = [], []
        for i in range(len(data) - self.total_window_size + 1):
            X.append(data[i:i + self.input_width, :-1])  # All except the last column (input)
            Y.append(data[i + self.input_width - self.label_width:i + self.input_width, -1:])  # Last column (label)

        print(f'Generated {len(X)} windows of shape {X[0].shape} for inputs and {Y[0].shape} for labels')
        return np.array(X), np.array(Y)

    def get_datasets(self, batch_size):
        train_X, train_Y = self.make_windows(self.train_df)
        val_X, val_Y = self.make_windows(self.val_df)
        test_X, test_Y = self.make_windows(self.test_df)

        # Ensure dimensions are correct
        assert train_X.shape[1] == self.input_width, "Input width mismatch"
        assert train_Y.shape[1] == self.label_width, "Label width mismatch"

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


def main():
    # Parameters
    file_path = './UV_cured_PBD-3_ca-6min.xlsx'
    INPUT_WIDTH = 4
    LABEL_WIDTH = 2
    SHIFT = 1
    BATCH_SIZE = 16

    # Load data
    train_df, val_df, test_df = load_data(file_path)

    # Instantiate the window generator
    window_gen = WindowGenerator(input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=SHIFT,
                                 train_df=train_df, val_df=val_df, test_df=test_df)

    print(f"\n{window_gen}")

    train_loader, val_loader, test_loader = window_gen.get_datasets(batch_size=BATCH_SIZE)

    for batch_X, batch_Y in train_loader:
        print(f'Window shape (batch, time, features): {batch_X.shape}')
        print(f'Inputs shape (batch, time, features): {batch_X.shape}')
        print(f'Labels shape (batch, time, features): {batch_Y.shape}')

        # Convert the tensors to numpy for easy readability
        inputs_np = batch_X.numpy()
        labels_np = batch_Y.numpy()

        # Iterate over the batch and print input sequences and corresponding labels
        for i in range(len(inputs_np)):  # Iterate over each sequence in the batch
            print(f'\nSequence {i + 1}:')
            print(f'Input sequence: {inputs_np[i]}')  # Print input sequence
            print(f'Label sequence: {labels_np[i]}')  # Print corresponding label sequence

        break  # Only print the first batch and exit


if __name__ == "__main__":
    main()
