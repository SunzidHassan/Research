from torch import nn

class ConvDenseModelv2(nn.Module):
    def __init__(self, num_features, conv_width, filters, dense_units, out_steps, dropout_rate=0.3):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
        nn.Conv1d(
            in_channels=num_features,  # Matches the number of input features
            out_channels=filters,
            kernel_size=conv_width,
            padding=(conv_width // 2)  # Same padding for sequence preservation
        ),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.Conv1d(
            in_channels=filters,
            out_channels=filters // 2,
            kernel_size=conv_width,
            padding=(conv_width // 2)
        ),
        nn.BatchNorm1d(filters // 2),
        nn.ReLU(),
        # Global average pooling
        nn.AdaptiveAvgPool1d(1)
        )

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters // 2, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            nn.Linear(dense_units, dense_units * 2),
            nn.BatchNorm1d(dense_units * 2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            nn.Linear(dense_units * 2, dense_units * 2),
            nn.BatchNorm1d(dense_units * 2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            nn.Linear(dense_units * 2, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            nn.Linear(dense_units, out_steps)
        )

    def forward(self, inputs):
        # Permute inputs to match Conv1D's expected input shape: (batch_size, features, time_steps)
        x = inputs.permute(0, 2, 1)  # (batch_size, num_features, time_steps)

        # Apply convolutional block 1
        return self.regression(self.conv_block_1(x))

