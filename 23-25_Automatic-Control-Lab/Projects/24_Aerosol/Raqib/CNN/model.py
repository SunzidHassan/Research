import torch
import torch.nn as nn


class ConvDenseModel(nn.Module):
    def __init__(self, num_features, conv_width, filters, dense_units, out_steps, dropout_rate=0.3):
        super(ConvDenseModel, self).__init__()

        # Convolutional block 1
        self.conv1 = nn.Conv1d(
            in_channels=num_features,  # Matches the number of input features
            out_channels=filters,
            kernel_size=conv_width,
            padding=(conv_width // 2)  # Same padding for sequence preservation
        )
        self.batch_norm1 = nn.BatchNorm1d(filters)
        self.relu1 = nn.ReLU()

        # Convolutional block 2
        self.conv2 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters // 2,
            kernel_size=conv_width,
            padding=(conv_width // 2)
        )
        self.batch_norm2 = nn.BatchNorm1d(filters // 2)
        self.relu2 = nn.ReLU()

        # Convolutional block 3
        self.conv3 = nn.Conv1d(
            in_channels=filters // 2,
            out_channels=filters // 2,
            kernel_size=conv_width,
            padding=(conv_width // 2)
        )
        self.batch_norm3 = nn.BatchNorm1d(filters // 2)
        self.relu3 = nn.ReLU()

        # Convolutional block 4
        self.conv4 = nn.Conv1d(
            in_channels=filters // 2,
            out_channels=filters // 4,
            kernel_size=conv_width,
            padding=(conv_width // 2)
        )
        self.batch_norm4 = nn.BatchNorm1d(filters // 4)
        self.relu4 = nn.ReLU()

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Pools to size (batch_size, filters // 8, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(filters // 4, dense_units)  # First dense layer
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after dense layer
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, out_steps)  # Final dense layer for output

    def forward(self, inputs):
        # Permute inputs to match Conv1D's expected input shape: (batch_size, features, time_steps)
        x = inputs.permute(0, 2, 1)  # (batch_size, num_features, time_steps)

        # Apply convolutional block 1
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        # Apply convolutional block 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        # Apply convolutional block 3
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        # Apply convolutional block 4
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)

        # Global average pooling
        x = self.global_avg_pool(x).squeeze(-1)  # Reduce sequence dimension

        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu_fc1(x)
        output = self.fc2(x)

        return output

