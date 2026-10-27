import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    """
    Multi-Layer Perceptron used for final power prediction.
    """

    def __init__(
        self,
        input_dim=64,
        hidden_dim=32,
        output_dim=1,
        dropout=0.2
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x :
            Shape -> (num_nodes, input_dim)
            or
            Shape -> (batch_size, num_nodes, input_dim)
        """

        if x.dim() == 2:
            return self.network(x)

        batch_size = x.size(0)
        num_nodes = x.size(1)

        x = x.reshape(-1, x.size(-1))

        x = self.network(x)

        x = x.reshape(
            batch_size,
            num_nodes,
            -1
        )

        return x.squeeze(-1)