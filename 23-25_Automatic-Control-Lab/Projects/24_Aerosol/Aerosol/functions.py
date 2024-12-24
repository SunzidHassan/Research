# Import PyTorch
import torch
from torch import nn
import torch.nn.functional as F

# Dataloader
from torch.utils.data import DataLoader, TensorDataset, Subset

# Import matplotlib
import matplotlib.pyplot as plt

# Metrics
from torchmetrics import MeanSquaredError

# Timer
from timeit import default_timer as timer

# Progress bar
from tqdm import tqdm

# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv
import h5py
import glob
import os
from pathlib import Path

import datetime

# train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import time

import datetime
import pathlib

# Cherck versions
print(torch.__version__)


class WindowGenerator:
    def __init__(self,
                window_size: int,
                label_size: int,
                shift: int,
                X_train: pd.DataFrame,
                y_train: pd.DataFrame,
                X_valid: pd.DataFrame,
                y_valid: pd.DataFrame,
                X_test: pd.DataFrame,
                y_test: pd.DataFrame,
                batch_size: int = 128):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.window_size = window_size
        self.label_size = label_size
        self.shift = shift
        self.batch_size = batch_size

        self.total_window_size = self.window_size + self.shift
        self.input_slice = slice(0, self.window_size)
        self.label_start = self.window_size - self.label_size
        self.label_slice = slice(self.label_start, None)

        self.input_columns = X_train.columns
        self.label_columns = y_train.columns


    def __repr__(self):
        input_indices = list(range(self.window_size))
        label_indices = list(range(self.label_start, self.total_window_size-1))
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {input_indices}',
            f'Label indices: {label_indices}',
            f'Input column name(s): {list(self.input_columns)}',
            f'Label column name(s): {list(self.label_columns)}'
        ])


    def make_windows(self,
                     X_df: pd.DataFrame,
                     y_df: pd.DataFrame):
        X_data = np.array(X_df, dtype=np.float32)
        y_data = np.array(y_df, dtype=np.float32)
        X, y = [], []
        for i in range(len(X_data) - self.total_window_size + 1):
            X.append(X_data[i: i + self.window_size])
            y.append(y_data[i + self.window_size])
        
        print(f'Generated {len(X)} windows of shape {X[0].shape} for inputs and {y[0].shape} for labels')
        return np.array(X), np.array(y)
    

    def get_datasets(self, batch_size: int):
        train_X, train_y = self.make_windows(self.X_train, self.y_train)
        valid_X, valid_y = self.make_windows(self.X_valid, self.y_valid)
        test_X, test_y = self.make_windows(self.X_test, self.y_test)

        train_dataset = TensorDataset(
            torch.tensor(train_X),
            torch.tensor(train_y)
        )
        valid_dataset = TensorDataset(
            torch.tensor(valid_X),
            torch.tensor(valid_y)
        )
        test_dataset = TensorDataset(
            torch.tensor(test_X),
            torch.tensor(test_y)
        )

        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, valid_dataloader, test_dataloader


def load_data(X_train, y_train, X_test, y_test):
    X_df = pd.read_csv(X_train)
    y_df = pd.read_csv(y_train)
    n = len(X_df)
    X_train = X_df[0:int(n*0.8)]
    y_train = y_df[0:int(n*0.8)]
    X_valid = X_df[int(n*0.8):int(n*0.9)]
    y_valid = y_df[int(n*0.8):int(n*0.9)]
    X_test = X_df[int(n*0.9):int(n*0.9)+int(config['batch_size'])]
    y_test = y_df[int(n*0.9):int(n*0.9)+int(config['batch_size'])]
    # X_test = pd.read_csv(X_test)
    # y_test = pd.read_csv(y_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, input, target):
        # Create the weight tensor with the required weights
        # weights = torch.tensor([1.3228e+01, 2.9112e+00, 3.2229e+00, 3.8279e+00, 4.7273e+00, 6.5284e+00,
        # 1.1661e+01, 1.1570e+01, 3.7472e+01, 9.2387e+01, 1.0000e+05, 1.0073e+00,
        # 1.0959e+00, 1.0744e+00, 1.0479e+00, 1.0221e+00, 1.0053e+00, 9.9999e-01,
        # 1.0072e+00, 1.0400e+00, 1.1234e+00, 1.2876e+00]).to(input.device)

        # weights = torch.tensor([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
        #                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).to(input.device)
        weights = torch.tensor([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).to(input.device)
        # weights = torch.tensor([10, 3, 3, 3, 4, 6, 10, 9, 20, 30, 35,
        #                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).to(input.device)

        # weights = torch.cat((torch.full((11,), 10.0), torch.full((11,), 1.0)), dim=0).to(input.device)
        # Compute the weighted MSE
        return torch.mean(weights * (input - target) ** 2)


def train_model(model, train_dataloader, valid_dataloader, num_epochs, learning_rate, weight_decay, device):
    loss_fn = WeightedMSE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_loss = float('inf')
    train_loss_history = []
    early_stop_count = 0
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_dataloader, position=0)

        for X, y in train_pbar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_record.append(loss.item())
            
            train_pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        train_loss_history.append(mean_train_loss)

        model.eval()
        loss_record = []
        with torch.inference_mode():
            for X, y in valid_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                loss_record.append(loss.item())

        scheduler.step()
        mean_valid_loss = sum(loss_record) / len(loss_record)

        print(f"Train loss: {mean_train_loss:.5f} | Valid loss: {mean_valid_loss:.5f}\n")

        # Save the best model
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss

            # Generate timestamped model save path
            now = datetime.datetime.now()
            MODEL_NAME = f"{now.month}_{now.day}_{model.__class__.__name__}.pth"
            MODEL_SAVE_PATH = pathlib.Path(config['save_path']).parent / MODEL_NAME

            try:
                # Create directory if it does not exist
                MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"Saving model to: {MODEL_SAVE_PATH} with loss: {best_loss:.5f}")
            except Exception as e:
                print(f"Error saving model: {e}")

            early_stop_count = 0
        else:
            early_stop_count += 1

        # Early stopping
        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, halting the training session.')
            break


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               input_scaler: StandardScaler,
               output_scaler: StandardScaler,
               start_time: float,
               end_time: float,
               device: torch.device):
    """
    Evaluation loop with normalization and denormalization.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        input_scaler (StandardScaler): Scaler used for input normalization.
        output_scaler (StandardScaler): Scaler used for output normalization.
        device (torch.device): Device to use for evaluation.

    Returns:
        dict: Evaluation metrics including loss and accuracy.
    """
    loss_fn = torch.nn.MSELoss()  # Example loss function
    total_loss, total_samples = 0.0, 0

    model.eval()
    predictions, true_labels = [], []

    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device)

            # Forward pass
            pred = model(X)

            # Compute loss
            loss = loss_fn(pred, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

            # Store predictions and true labels
            predictions.append(pred.cpu())
            true_labels.append(y.cpu())

    # Calculate average loss
    avg_loss = total_loss / total_samples
    train_time = end_time - start_time

    # Convert to tensors for plotting
    predictions = torch.cat(predictions, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    print(f"Prediction shape: {predictions.shape}")
    print(f"True label shape: {true_labels.shape}")
    
    # Plot feature predictions
    num_features = predictions.size(1)
    fig1, axes1 = plt.subplots(1, num_features, figsize=(5 * num_features, 5), sharey=True)
    # print(num_features)

    xAxis = torch.arange(1, predictions.size(0) + 1)

    for i in range(22):
        axes1[i].scatter(xAxis, predictions[:, i], label='Prediction', color='blue', alpha=0.6)
        axes1[i].scatter(xAxis, true_labels[:, i], label='Actual', color='orange', alpha=0.6)
        axes1[i].set_title(f'Feature {i+1}')
        axes1[i].legend()
        axes1[i].set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.show()

    # Plot prediction vs actual values with diagonal line
    fig2, axes2 = plt.subplots(1, 11, figsize=(5 * 11, 5), sharey=True)

    for i in range(11):
        axes2[i].scatter(predictions[:, i], true_labels[:, i], label='Prediction vs True', alpha=0.6)
        axes2[i].set_title(f'Feature {i+1}')
        axes2[i].set_xlabel("Predictions")
        axes2[i].set_ylabel("True Values")
        # Add diagonal line y = x
        # plt.axline((0, 0), slope=1, color='red', linestyle='--', label='y=x')
        min_val = min(predictions[:, i].min(), true_labels[:, i].min())
        max_val = max(predictions[:, i].max(), true_labels[:, i].max())
        axes2[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
        axes2[i].legend()
        axes2[i].set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.show()

    # Plot another set of features with diagonal line
    fig3, axes3 = plt.subplots(1, 11, figsize=(5 * 11, 5), sharey=True)

    for i in range(11):
        axes3[i].scatter(predictions[:, i+11], true_labels[:, i+11], label='Prediction vs True', alpha=0.6)
        axes3[i].set_title(f'Feature {i+12}')
        axes3[i].set_xlabel("Predictions")
        axes3[i].set_ylabel("True Values")
        # Add diagonal line y = x
        # axes3[i]((0, 0), slope=1, color='red', linestyle='--', label='y=x')
        min_val = min(predictions[:, i+11].min(), true_labels[:, i+11].min())
        max_val = max(predictions[:, i+11].max(), true_labels[:, i+11].max())
        axes3[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
        axes3[i].legend()
        axes3[i].set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.show()

    return {
        "model_name": model.__class__.__name__,
        "model_loss": round(total_loss, 5),
        "train_time": round(train_time, 2)
    }

