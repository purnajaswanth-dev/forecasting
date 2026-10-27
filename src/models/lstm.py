import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    LSTM encoder for pilot PV units.

    Input:
        (batch_size, num_pilot_units, sequence_length)

    Output:
        (batch_size, num_pilot_units, hidden_dim)
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        """
        x:
            (batch, pilots, seq_len)
        """

        batch_size = x.size(0)
        num_pilots = x.size(1)
        seq_len = x.size(2)

        # Merge pilot dimension into batch
        x = x.reshape(
            batch_size * num_pilots,
            seq_len,
            1
        )

        output, (hidden, cell) = self.lstm(x)

        embedding = hidden[-1]

        embedding = embedding.reshape(
            batch_size,
            num_pilots,
            self.hidden_size
        )

        return embedding