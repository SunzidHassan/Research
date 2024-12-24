# Import PyTorch
import torch

# Timer
from timeit import default_timer as timer


# Reading/Writing Data
import pandas as pd
import datetime

# train_test_split
from sklearn.preprocessing import StandardScaler

import time

import datetime
import pathlib

from functions import load_data, WindowGenerator, train_model, eval_model
from model import LSTMModel

config = {
    'seed': 42,
    'X_train': 'data/X_train.csv',
    'y_train': 'data/y_train.csv',
    'X_test': 'data/X_test.csv',
    'y_test': 'data/y_test.csv',
    'n_epochs': 6,
    'batch_size': 128,
    'learning_rate': 1e-4,
    'dropout': 0.3,
    'weight_decay': 1e-2,
    'early_stop': 2,
    'hidden_layers': 3,
    'hidden_units': 128,
    'kernel_size': 3,
    'filters': 128,
    'save_path': './models/dec22.pth',  # Your model will be saved here.
    'window_size': 4,
    'label_size': 1,
    'shift': 1
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    X_train = config['X_train']
    y_train = config['y_train']
    X_test = config['X_test']
    y_test = config['y_test']
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(X_train, y_train, X_test, y_test)

    WINDOW_SIZE = config['window_size']
    LABEL_SIZE = config['label_size']
    SHIFT = config['shift']
    BATCH_SIZE = config['batch_size']

    # Normalization
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    input_scaler.fit(X_train)
    output_scaler.fit(y_train)

    X_train = pd.DataFrame(input_scaler.transform(X_train))
    y_train = pd.DataFrame(output_scaler.transform(y_train))
    X_valid = pd.DataFrame(input_scaler.transform(X_valid))
    y_valid = pd.DataFrame(output_scaler.transform(y_valid))
    X_test = pd.DataFrame(input_scaler.transform(X_test))
    y_test = pd.DataFrame(output_scaler.transform(y_test))

    window_gen = WindowGenerator(window_size = WINDOW_SIZE,
                                    label_size = LABEL_SIZE,
                                    shift = SHIFT,
                                    X_train = X_train,
                                    y_train = y_train,
                                    X_valid = X_valid,
                                    y_valid = y_valid,
                                    X_test = X_test,
                                    y_test = y_test,
                                    batch_size = BATCH_SIZE)


    train_dataloader, valid_dataloader, test_dataloader = window_gen.get_datasets(batch_size=BATCH_SIZE)

    # FNN Settings
    INPUT_DIM = len(X_train.columns)
    HIDDEN_DIM = config['hidden_units']
    OUTPUT_DIM = 22
    DROPOUT_RATE = config['dropout']
    HIDDEN_LAYERS = config['hidden_layers']

    # Initialize the model
    model = LSTMModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, HIDDEN_LAYERS, DROPOUT_RATE).to(device)

    NUM_EPOCHS = config['n_epochs']
    LR = config['learning_rate']
    WD = config['weight_decay']

    train_time_start = time.perf_counter()
    train_model(model,
                train_dataloader,
                valid_dataloader,
                NUM_EPOCHS,
                LR,
                WD,
                device=device)
    train_time_end = time.perf_counter()

    now = datetime.datetime.now()
    # MODEL_NAME = f"{now.month}_{'22'}_{model.__class__.__name__}.pth"
    MODEL_NAME = f"{now.month}_{now.day}_{model.__class__.__name__}.pth"
    MODEL_SAVE_PATH = pathlib.Path(config['save_path']).parent / MODEL_NAME

    print("Loading best model for evaluation.")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)

    fnn_model_v1_result = eval_model(model=model,
                                    data_loader=test_dataloader,
                                    input_scaler=input_scaler,
                                    output_scaler=output_scaler,
                                    start_time=train_time_start,
                                    end_time=train_time_end,
                                    device=device)
    fnn_model_v1_result

if __name__ == "__main__":
    main()
