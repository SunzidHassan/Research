import torch
from data import WindowGenerator, load_data
from model import FeedForwardNN
from training import train_model, evaluate_model
from visualization import plot_real_vs_pred

def main():
    file_path = './UV_cured_PBD-3_ca-6min.xlsx'
    train_df, val_df, test_df = load_data(file_path)

    # Normalization directly in the code without saving/loading
    train_mean = train_df.mean()
    train_std = train_df.std()

    # Normalize the data using the mean and std of the training data
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    input_width = 200  # Window size
    label_width = 1  # Predict the last step
    shift = 1
    batch_size = 128

    window_gen = WindowGenerator(input_width, label_width, shift, train_df, val_df, test_df)

    train_loader, val_loader, test_loader = window_gen.get_datasets(batch_size=batch_size)

    # FNN settings
    input_dim = input_width * (len(train_df.columns) - 1)
    hidden_dim = 256
    output_dim = label_width
    dropout_rate = 0.3

    model = FeedForwardNN(input_dim, hidden_dim, output_dim, dropout_rate)

    num_epochs = 150
    learning_rate = 0.00001
    weight_decay = 1e-2
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay)

    print("Loading best model for evaluation.")
    model.load_state_dict(torch.load('best_model.pth'))

    test_model(model, test_loader, train_mean, train_std)

def test_model(model, test_loader, train_mean, train_std):
    rmse, mape, pearson_corr, predicted, actual = evaluate_model(model, test_loader)

    # Denormalize the predictions and actual values
    predicted = predicted * train_std.iloc[-1] + train_mean.iloc[-1]
    actual = actual * train_std.iloc[-1] + train_mean.iloc[-1]

    print(f"RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Pearson Correlation: {pearson_corr:.4f}")

    time = test_loader.dataset.tensors[0][:, -1, 0].numpy()

    min_length = min(len(time), len(actual), len(predicted))
    time = time[:min_length]
    actual = actual[:min_length]
    predicted = predicted[:min_length]

    plot_real_vs_pred(actual, predicted, time)

if __name__ == "__main__":
    main()
