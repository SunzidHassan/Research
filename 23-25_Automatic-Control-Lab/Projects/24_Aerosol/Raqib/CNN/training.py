import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, device):
    """Train the model with properly permuted inputs."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_loss = float('inf')
    early_stop_patience = 30
    patience_counter = 0

    # Debugging: Print the input and label shapes for the first batch
    for inputs, labels in train_loader:
        print(f"Debug - Input shape: {inputs.shape}, Label shape: {labels.shape}")
        break

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # No need to permute here; handled in the model itself
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
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')


def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test set and compute RMSE, MAPE, and Pearson correlation."""
    criterion = torch.nn.MSELoss(reduction='mean')
    predicted_strain = []
    actual_strain = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)

            # Debugging: Print the shape of inputs and labels
            # print(f"Debug (Eval) - Input shape: {inputs.shape}, Label shape: {labels.shape}")

            # Pass inputs directly to the model without permutation
            predictions = model(inputs)

            # Collect predicted and actual values
            predicted_strain.extend(predictions.cpu().numpy())  # Move to CPU for evaluation
            actual_strain.extend(labels.cpu().numpy())  # Move to CPU for evaluation

    # Convert to numpy arrays and flatten if necessary
    predicted_strain = np.array(predicted_strain).reshape(-1)
    actual_strain = np.array(actual_strain).reshape(-1)

    # Compute evaluation metrics
    rmse = np.sqrt(criterion(torch.tensor(predicted_strain), torch.tensor(actual_strain)).item())
    mape = np.mean(np.abs((actual_strain - predicted_strain) / actual_strain)) * 100
    pearson_corr, _ = pearsonr(actual_strain, predicted_strain)  # Compute Pearson correlation

    return rmse, mape, pearson_corr, predicted_strain, actual_strain




