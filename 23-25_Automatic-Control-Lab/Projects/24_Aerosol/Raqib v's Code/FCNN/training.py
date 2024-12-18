import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_loss = float('inf')
    early_stop_patience = 30
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.view(inputs.size(0), -1), labels.view(labels.size(0), -1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.view(inputs.size(0), -1), labels.view(labels.size(0), -1)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')


def evaluate_model(model, test_loader, original_df=None):
    """Evaluate the model on the test set and compute RMSE, MAPE, and Pearson correlation."""
    criterion = torch.nn.MSELoss(reduction='mean')
    predicted_strain = []
    actual_strain = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten inputs
            predictions = model(inputs)

            # Collect predicted and actual values
            predicted_strain.extend(predictions.numpy())
            actual_strain.extend(labels.numpy())

    # Convert to numpy arrays and flatten if necessary
    predicted_strain = np.array(predicted_strain).reshape(-1)
    actual_strain = np.array(actual_strain).reshape(-1)

    rmse = np.sqrt(criterion(torch.tensor(predicted_strain), torch.tensor(actual_strain)).item())
    mape = np.mean(np.abs((actual_strain - predicted_strain) / actual_strain)) * 100
    pearson_corr, _ = pearsonr(actual_strain, predicted_strain)  # Compute Pearson correlation

    return rmse, mape, pearson_corr, predicted_strain, actual_strain

