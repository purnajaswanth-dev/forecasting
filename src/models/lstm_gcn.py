import torch
import torch.nn as nn

from .lstm import LSTMEncoder
from .gcn import GCNEncoder
from .mlp import MLPPredictor


class LSTMGCNModel(nn.Module):
    """
    LSTM -> GCN -> MLP

    Implements the architecture proposed in the paper.

    LSTM learns temporal embeddings for pilot units.

    GCN propagates those embeddings over the graph.

    MLP predicts power for every PV unit.
    """

    def __init__(
        self,
        num_nodes,
        pilot_indices,
        hidden_dim=64
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.pilot_indices = torch.LongTensor(pilot_indices)

        self.lstm = LSTMEncoder(
            input_size=1,
            hidden_size=hidden_dim
        )

        self.gcn = GCNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )

        self.mlp = MLPPredictor(
            input_dim=hidden_dim,
            hidden_dim=32,
            output_dim=1
        )

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None
    ):
        """
        Parameters
        ----------
        x

            Shape:
            (batch_size,
             num_pilot_units,
             sequence_length)

        Returns
        -------
        Shape:
            (batch_size,
             num_nodes)
        """

        device = x.device

        pilot_indices = self.pilot_indices.to(device)

        batch_size = x.size(0)

        pilot_embeddings = self.lstm(x)

        outputs = []

        for batch in range(batch_size):

            node_features = torch.zeros(
                self.num_nodes,
                self.hidden_dim,
                device=device
            )

            node_features[pilot_indices] = pilot_embeddings[batch]

            graph_embeddings = self.gcn(
                node_features,
                edge_index,
                edge_weight
            )

            prediction = self.mlp(
                graph_embeddings
            )

            outputs.append(prediction)

        outputs = torch.stack(outputs)

        return outputs