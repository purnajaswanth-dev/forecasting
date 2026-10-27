import torch
import torch.nn as nn


class SolarForecastModel(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            dropout,
            output_size
    ):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout

        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(

            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)

        )

        self._initialize_weights()

    def _initialize_weights(self):

        for name, param in self.lstm.named_parameters():

            if "weight" in name:
                nn.init.xavier_uniform_(param)

            elif "bias" in name:
                nn.init.zeros_(param)

        for layer in self.fc:

            if isinstance(layer, nn.Linear):

                nn.init.kaiming_uniform_(
                    layer.weight,
                    nonlinearity="relu"
                )

                nn.init.zeros_(layer.bias)

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        hidden = hidden[-1]

        hidden = self.dropout(hidden)

        prediction = self.fc(hidden)

        return prediction.squeeze()