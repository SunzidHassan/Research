import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Pass through LSTM layer
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch_size, seq_length, hidden_dim)

        # Use only the last time step's output
        last_time_step = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Pass through fully connected layer
        output = self.fc(last_time_step)  # Shape: (batch_size, output_dim)

        return output
