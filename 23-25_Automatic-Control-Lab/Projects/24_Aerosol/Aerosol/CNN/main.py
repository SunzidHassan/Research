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
from model import ConvDenseModelv2

from config import config

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
    INPUT_DIM = WINDOW_SIZE * len(X_train.columns)
    HIDDEN_DIM = config['hidden_units']
    OUTPUT_DIM = 22
    DROPOUT_RATE = config['dropout']

    
    # ConvDenseModel settings
    NUM_FEATURES = len(X_train.columns)  # Number of input features
    KERNEL_SIZE = config['kernel_size']  # Fixed kernel size (3, 5, or 7)
    FILTERS = config['filters']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the model
    model = ConvDenseModelv2(NUM_FEATURES, KERNEL_SIZE, FILTERS, HIDDEN_DIM, OUTPUT_DIM, DROPOUT_RATE).to(device)

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

    fnn_model_v2_result = eval_model(model=model,
                                    data_loader=test_dataloader,
                                    input_scaler=input_scaler,
                                    output_scaler=output_scaler,
                                    start_time=train_time_start,
                                    end_time=train_time_end,
                                    device=device)
    fnn_model_v2_result

if __name__ == "__main__":
    main()
