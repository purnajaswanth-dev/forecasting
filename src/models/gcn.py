import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """
    Graph Convolution Network.

    Receives LSTM embeddings of pilot nodes,
    propagates information through the graph,
    and produces node embeddings for every PV unit.
    """

    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        dropout=0.2
    ):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None
    ):
        """
        Parameters
        ----------
        x :
            (num_nodes, input_dim)

        edge_index :
            Graph connectivity.

        edge_weight :
            Edge weights.
        """

        x = self.conv1(
            x,
            edge_index,
            edge_weight
        )

        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
            edge_weight
        )

        return x