import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from data import WindowGenerator, load_data
from training import train_model, evaluate_model
from visualization import plot_real_vs_pred
from torch.utils.data import DataLoader, TensorDataset
from model import ConvDenseModel

def main():
    # File path for the dataset
    file_path = './UV_cured_PBD-3_ca-6min.xlsx'
    train_df, val_df, test_df = load_data(file_path)

    # Normalization directly in the code without saving/loading
    train_mean = train_df.mean()
    train_std = train_df.std()

    # Normalize the data using the mean and std of the training data
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # Set flexible input and label window sizes
    input_width = 100  # Changeable window size for inputs
    label_width = 1    # Changeable window size for labels
    shift = 1          # Predict the next step
    batch_size = 8

    # Initialize WindowGenerator with flexible settings
    window_gen = WindowGenerator(input_width, label_width, shift, train_df, val_df, test_df)
    train_loader, val_loader, test_loader = window_gen.get_datasets(batch_size=batch_size)

    # ConvDenseModel settings
    num_features = len(train_df.columns) - 1  # Number of input features
    kernel_size = 3  # Fixed kernel size (3, 5, or 7)
    filters = 128
    dense_units = 128
    out_steps = label_width  # Dynamically matches the label width
    dropout_rate = 0.3

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = ConvDenseModel(num_features, kernel_size, filters, dense_units, out_steps, dropout_rate).to(device)

    # Training settings
    num_epochs = 100
    learning_rate = 0.0001
    weight_decay = 1e-4

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, device)

    # Save the trained model
    torch.save(model.state_dict(), 'conv_dense_model.pth')
    print("Model saved successfully as 'conv_dense_model.pth'")

    # Evaluate the model on the test set
    test_model(model, test_loader, train_mean, train_std, device)


def test_model(model, test_loader, train_mean, train_std, device):
    """Evaluate the model on the test set."""
    model.to(device)  # Ensure the model is on the correct device
    model.eval()
    rmse, mape, pearson_corr, predicted, actual = evaluate_model(model, test_loader, device)

    # Denormalize the predictions and actual values for proper evaluation
    predicted = predicted * train_std.iloc[-1] + train_mean.iloc[-1]
    actual = actual * train_std.iloc[-1] + train_mean.iloc[-1]

    print(f"RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Pearson Correlation: {pearson_corr:.4f}")

    # Extract time values corresponding to the actual and predicted values
    time = test_loader.dataset.tensors[0][:, -1, 0].cpu().numpy()  # Extract the last time step of each sequence

    # Ensure time, actual, and predicted have the same length
    min_length = min(len(time), len(actual), len(predicted))
    time = time[:min_length]
    actual = actual[:min_length]
    predicted = predicted[:min_length]

    # Plot the results
    plot_real_vs_pred(actual, predicted, time)


def test_on_new_data():
    # File paths for training and new datasets
    file_path = './UV_cured_PBD-3_ca-6min.xlsx'
    new_file_path = './UV_cured_PBD-3_ ca-140s.xlsx'

    # Load and normalize data
    train_df, val_df, test_df = load_data(file_path)
    new_df = pd.read_excel(new_file_path)

    train_mean = train_df.mean()
    train_std = train_df.std()

    # Normalize the data using the mean and std of the training data
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    new_df = (new_df - train_mean) / train_std

    # Set flexible input and label window sizes
    input_width = 100  # Changeable input window size
    label_width = 1   # Changeable label window size
    shift = 1
    batch_size = 8

    # Create window generators for the new dataset
    window_gen = WindowGenerator(input_width, label_width, shift, train_df, val_df, test_df)
    new_X, new_Y = window_gen.make_windows(new_df)
    new_dataset = TensorDataset(torch.tensor(new_X), torch.tensor(new_Y))
    new_loader = DataLoader(new_dataset, batch_size=batch_size)

    # ConvDenseModel settings
    num_features = len(train_df.columns) - 1
    kernel_size = 3  # Fixed kernel size (3, 5, or 7)
    filters = 128
    dense_units = 128
    out_steps = label_width
    dropout_rate = 0.3

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvDenseModel(num_features, kernel_size, filters, dense_units, out_steps, dropout_rate).to(device)
    model.load_state_dict(torch.load('conv_dense_model.pth', map_location=device))
    print("Model loaded successfully from 'conv_dense_model.pth'")

    # Evaluate the model on the new dataset
    test_model_new_data(model, new_loader, train_mean, train_std, device)


def test_model_new_data(model, test_loader, train_mean, train_std, device):
    """Evaluate the model on the new dataset and compute metrics."""
    model.eval()
    predicted_strain = []
    actual_strain = []
    time_steps = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the same device
            inputs, labels = inputs.to(device), labels.to(device)

            # Collect the corresponding time values for the current batch
            time_steps_batch = inputs[:, :, 0].cpu().numpy()

            # Make predictions
            predictions = model(inputs)

            # Append to the result lists
            predicted_strain.extend(predictions.cpu().numpy())
            actual_strain.extend(labels.cpu().numpy())
            time_steps.extend(time_steps_batch[:, -1])  # Take the last time step for each window

    # Convert to numpy arrays
    predicted_strain = np.array(predicted_strain).flatten()
    actual_strain = np.array(actual_strain).flatten()
    time_steps = np.array(time_steps)

    # Denormalize the predictions and actual values
    predicted_strain = predicted_strain * train_std.iloc[-1] + train_mean.iloc[-1]
    actual_strain = actual_strain * train_std.iloc[-1] + train_mean.iloc[-1]

    # Ensure alignment between time, actual, and predicted
    min_length = min(len(time_steps), len(actual_strain), len(predicted_strain))
    time_steps = time_steps[:min_length]
    actual_strain = actual_strain[:min_length]
    predicted_strain = predicted_strain[:min_length]

    # Compute evaluation metrics
    rmse = np.sqrt(np.mean((actual_strain - predicted_strain) ** 2))
    mape = np.mean(np.abs((actual_strain - predicted_strain) / actual_strain)) * 100
    if len(actual_strain) > 1:  # Ensure enough data points for correlation
        pearson_corr, _ = pearsonr(actual_strain, predicted_strain)
    else:
        pearson_corr = float('nan')

    # Print the metrics
    print(f"RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Pearson Correlation: {pearson_corr:.4f}")

    # Save results for inspection
    results_df = pd.DataFrame({
        'Time (min)': time_steps,
        'Actual Strain (%)': actual_strain,
        'Predicted Strain (%)': predicted_strain
    })
    results_df.to_csv('conv_dense_predicted_results.csv', index=False)
    print("Results saved to 'conv_dense_predicted_results.csv'")

    # Plot the results
    plot_real_vs_pred(actual_strain, predicted_strain, time_steps)


if __name__ == "__main__":
    # main()
    test_on_new_data()
